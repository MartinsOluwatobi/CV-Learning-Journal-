import torch
from torchvision.ops import box_iou
from LossFunction import box_coordinate
def LabelAccuracy(cls_logit,label):
    cls_label = torch.argmax(cls_logit, dim = 1)
    accuracy = (cls_label==label).sum().item() / len(label)
    return accuracy
def BoxAccuracy(pred_delta, target_delta, proposal):
    pred_boxes = box_coordinate(pred_delta, proposal)
    pred_boxes[:, 2:] = torch.max(pred_boxes[:, 2:], pred_boxes[:, :2] + 1e-3)
    gt_boxes = box_coordinate(target_delta,proposal)
    iou_matrix = box_iou(pred_boxes,gt_boxes)
    accuracy = torch.diag(iou_matrix).mean().item()
    return accuracy