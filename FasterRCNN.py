from torchvision.models import resnet50, ResNet50_Weights
from torch import nn 
import torch
from torchvision.ops import nms, boxes
import torch.nn.functional as F
from Backbone import FeatureExtractor
from RegionProposalNetwork import RegionProposalNetwork
from RoiPooling import RoIPooling
''''Image -> Backbone (ResnNet) -> Feature Map ->Region Proposal Network
-> RoI Pooling -> Classifier + Bounding Box Regressor'''

class FasterRCNN(nn.Module):
    def __init__ (self, num_classes, image_shape, in_channels= 2048, mid_channels= 512, roi_output_size = (7,7)):
        super().__init__()
        self.feature_extractor = FeatureExtractor() # returns feature map of shape batch size ,1024, h,w
        self.rpn = RegionProposalNetwork(in_channels, mid_channels, image_shape) 
        self.roi_pooling = RoIPooling((roi_output_size)) 
        flat_dim = in_channels * roi_output_size[0] * roi_output_size[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim , 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim , 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes * 4)
        )

    def forward (self,x):
        feature_map = self.feature_extractor(x) # B, 2048 ,7,7
        proposal_boxes,rpn_reg_layer, rpn_cls, anchors, batch_idx = self.rpn(feature_map)
        rois_with_idx = torch.cat([batch_idx.unsqueeze(1).float(),proposal_boxes], dim=1)
        pooled_features = self.roi_pooling(feature_map, rois_with_idx)
        pooled_features = pooled_features.squeeze(1)
        class_logits = self.classifier(pooled_features)
        bbox_deltas = self.bbox_regressor(pooled_features)

        return class_logits, bbox_deltas


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = FasterRCNN(num_classes=21, image_shape=(800, 800))
    dummy = torch.randn(1, 3, 800, 800)
    cls_scores, bbox_deltas, rpn_cls, proposals = model(dummy)
    print("cls_scores :", cls_scores.shape)
    print("bbox_deltas:", bbox_deltas.shape)
    print("rpn_cls    :", rpn_cls.shape)
    print("proposals  :", proposals.shape)