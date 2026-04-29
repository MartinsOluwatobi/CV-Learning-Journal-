import torch
import torch.nn.functional as F
from torchvision.ops import box_iou 
def Encode_rpn_target_and_label (anchors, ground_truth_boxes, neg_iou =0.3, positive_iou = 0.7):
    '''Group anchor boxes based on using intercept over union (iou)
      threshold when they compare with ground truth boxes
      1 is assigned to anchor box iou greater than the higher limit threshold 
      0 is assigned to anchor box with iou lesser than the lower limit threshold'''
    N = anchors.shape[0]
    # create label list of the size as anchor generated per image with value of -1. -1 means ignore 
    labels = torch.full((N,),-1, dtype= torch.long, device = anchors.device)
    # initiate target for the answers as zero 
    target = torch.zeros((N,4),dtype= torch.float32, device= anchors.device)
    if ground_truth_boxes.numel() == 0:
        labels[:] = 0
        return labels, target 
    # Get the iou matrix that compares each anchor box to the groundtruths 
    iou = box_iou(anchors, ground_truth_boxes) # Returns matrix of shape (N,M) M is the number if groundtruth for the image
    # For each anchor, get the ground truth box it overlaps most with
    anchor_max_iou, anchor_matched_idx = iou.max(dim=1)
    # For each ground truth box, get the anchor with the highest IoU
    ground_truth_max, ground_truth_max_index = iou.max(dim=0)
    #force assign anchor to ground truth if the best iou is less than the postive iou     
    best_anchor_per_ground_truth = (iou == ground_truth_max).any(dim=1)

    
    labels[anchor_max_iou>positive_iou] = 1
    labels[anchor_max_iou< neg_iou] = 0
    labels[best_anchor_per_ground_truth] = 1

    #get the positions of labels assigned values of 1 and 0
    pos_idx = (labels==1).nonzero(as_tuple=True)[0]
    neg_idx = (labels==0).nonzero(as_tuple=True)[0]

    # setting the amount of positive and negative labels that will be picked to avoid having imbalance labels
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

def Encode_final_target_label (proposal,ground_truth_box,ground_truth_labels, pos_threshold = 0.7,neg_threshold = 0.3, num_samples = 128, pos_fraction =0.25):
    N = proposal.shape[0]
    labels = torch.zeros((N), dtype=torch.long,device = proposal.device)
    iou = box_iou(proposal,ground_truth_box)
    max_iou, best_gt_idx = iou.max(dim = 1)
    labels[(max_iou>= neg_threshold) & (max_iou<=pos_threshold)]= -1
    labels[max_iou>= pos_threshold] = ground_truth_labels[best_gt_idx[max_iou>=pos_threshold]]

    pos_idx = (labels> 0).nonzero(as_tuple=True)[0]
    neg_idx = (labels==0).nonzero(as_tuple=True)[0]

    pos_idx = pos_idx[torch.randperm(len(pos_idx),device = proposal.device)[:int(num_samples * pos_fraction)]]
    neg_idx = neg_idx[torch.randperm(len(neg_idx),device = proposal.device)[:(num_samples - len(pos_idx))]]
    idx     = torch.cat([pos_idx, neg_idx])

    p, g    = proposal[idx], ground_truth_box[best_gt_idx[idx]]
    pw      = (p[:, 2] - p[:, 0]).clamp(1e-6)
    ph = (p[:, 3] - p[:, 1]).clamp(1e-6)
    gw      = (g[:, 2] - g[:, 0]).clamp(1e-6)
    gh = (g[:, 3] - g[:, 1]).clamp(1e-6)
    deltas  = torch.stack([
        (g[:, 0] + 0.5*gw - p[:, 0] - 0.5*pw) / pw,
        (g[:, 1] + 0.5*gh - p[:, 1] - 0.5*ph) / ph,
        torch.log(gw / pw),
        torch.log(gh / ph),
    ], dim=1)

    return idx, labels[idx], deltas

def rpn_loss(cls_logits, rpn_reg_layer, labels, target_deltas):
    mask = labels >= 0
    cls_loss = F.cross_entropy(cls_logits[mask], labels[mask])
    pos_mask = labels == 1
    if pos_mask.sum() > 0:
        reg_loss = F.smooth_l1_loss(
            rpn_reg_layer[pos_mask], 
            target_deltas[pos_mask], 
            beta=1.0/9.0 
        )
    else:
        reg_loss = torch.tensor(0.0).to(cls_logits.device)

    return cls_loss, reg_loss


def final_rcnn_loss(class_logits, box_deltas, labels, target_deltas):
    cls_loss = F.cross_entropy(class_logits, labels)
    pos_mask = labels > 0 
    if pos_mask.sum() > 0:
        pos_indices = torch.where(pos_mask)[0]
        pos_labels = labels[pos_mask]
        num_classes = box_deltas.shape[1] // 4
        pred_deltas = box_deltas.view(-1, num_classes, 4)
        pred_deltas = pred_deltas[pos_indices, pos_labels]

        reg_loss = F.smooth_l1_loss(pred_deltas, target_deltas[pos_mask])
    else:
        reg_loss = torch.tensor(0.0).to(class_logits.device)

    return cls_loss, reg_loss