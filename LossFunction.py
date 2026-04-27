import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

def Encode_rpn_target_and_label (anchors, ground_truth_boxes, neg_iou =0.3, positive_iou = 0.7):
    N = anchors.shape[0]
    labels = torch.full((N,),-1, dtype= torch.float32, device = anchors.device)
    target = torch.zeros((N,4),dtype= torch.float32, device= anchors.device)
    if ground_truth_boxes.numel() == 0:
        labels[:] = 0
        return labels, target
    iou = box_iou(anchors, ground_truth_boxes)
    anchor_max_iou, anchor_matched_idx = iou.max(dim=1)
    ground_truth_max, ground_truth_max_index = iou.max(dim=0)
    best_anchor_per_ground_truth = (iou == ground_truth_max).any(dim=1)

    labels[anchor_max_iou>positive_iou] = 1
    labels[anchor_max_iou< neg_iou] = 0
    labels[best_anchor_per_ground_truth] = 1

    pos_idx = (labels==1).nonzero(as_tuple=True)[0]
    neg_idx = (labels==0).nonzero(as_tuple=True)[0]

    num_pos = min(len(pos_idx),128)
    num_neg = min(len(neg_idx), 256 - num_pos)

    pos_idx = pos_idx[torch.randperm(len(pos_idx),device = anchors.device)][:num_pos]
    neg_idx = neg_idx[torch.randperm(len(neg_idx),device= anchors.device)][:num_neg]

    keep = torch.cat([pos_idx,neg_idx])
    mask = torch.zeros((N,), dtype= torch.bool, device = anchors.device)
    mask[keep] = True
    labels[~mask] = -1

    matched_gt = ground_truth_boxes[anchor_matched_idx]

    anchor_w   = anchors[:, 2] - anchors[:, 0]
    anchor_h   = anchors[:, 3] - anchors[:, 1]
    anchor_cx  = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy  = anchors[:, 1] + 0.5 * anchor_h

    gt_w       = matched_gt[:, 2] - matched_gt[:, 0]
    gt_h       = matched_gt[:, 3] - matched_gt[:, 1]
    gt_cx      = matched_gt[:, 0] + 0.5 * gt_w
    gt_cy      = matched_gt[:, 1] + 0.5 * gt_h

    dx = (gt_cx - anchor_cx) / anchor_w.clamp(min=1)
    dy = (gt_cy - anchor_cy) / anchor_h.clamp(min=1)
    dw = torch.log(gt_w.clamp(min=1) / anchor_w.clamp(min=1))
    dh = torch.log(gt_h.clamp(min=1) / anchor_h.clamp(min=1))

    deltas = torch.stack([dx, dy, dw, dh], dim=1)   # (N, 4)

    return labels, deltas



def rpn_loss(cls_logits, rpn_reg_layer, labels, target_deltas):
    """
    cls_logits: (N, 2) - Predicted Objectness
    reg_deltas: (N, 4) - Predicted offsets
    labels: (N) - Ground truth (1=fg, 0=bg, -1=ignore)
    target_deltas: (N, 4) - Actual offsets to GT
    """
    # 1. Classification Loss (Only for non-ignored anchors)
    # We use CrossEntropy only where label is not -1
    mask = labels >= 0
    cls_loss = F.cross_entropy(cls_logits[mask], labels[mask])

    # 2. Regression Loss (Only for Positive anchors)
    pos_mask = labels == 1
    if pos_mask.sum() > 0:
        # Smooth L1 is less sensitive to outliers than MSE
        reg_loss = F.smooth_l1_loss(
            rpn_reg_layer[pos_mask], 
            target_deltas[pos_mask], 
            beta=1.0/9.0 # Standard beta for RPN
        )
    else:
        reg_loss = torch.tensor(0.0).to(cls_logits.device)

    return cls_loss, reg_loss


def final_rcnn_loss(class_logits, box_deltas, labels, target_deltas):
    """
    class_logits: (N, num_classes)
    box_deltas: (N, num_classes * 4) - Faster R-CNN predicts 4 deltas per class
    labels: (N) - Ground truth class indices
    target_deltas: (N, 4)
    """
    # 1. Multi-class Classification Loss
    cls_loss = F.cross_entropy(class_logits, labels)

    # 2. Class-Specific Regression Loss
    # We only care about the deltas for the TRUE class
    pos_mask = labels > 0 # index 0 is usually background
    if pos_mask.sum() > 0:
        # Extract only the deltas corresponding to the ground truth class
        # (N, num_classes, 4) -> reshape to pick specific class
        pos_indices = torch.where(pos_mask)[0]
        pos_labels = labels[pos_mask]
        
        # Select the 4 coordinates for the specific class assigned to each RoI
        pred_deltas = box_deltas.view(-1, 2, 4)
        pred_deltas = pred_deltas[pos_indices, pos_labels]

        reg_loss = F.smooth_l1_loss(pred_deltas, target_deltas[pos_mask])
    else:
        reg_loss = torch.tensor(0.0).to(class_logits.device)

    return cls_loss, reg_loss