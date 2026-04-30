from torchvision.models import resnet50, ResNet50_Weights
from torch import nn 
import torch
from torchvision.ops import nms, boxes
import torch.nn.functional as F
from Backbone import FeatureExtractor
from RegionProposalNetwork import RegionProposalNetwork
from RoiPooling import RoIPooling

# Faster R-CNN architecture overview:
# 1. Feature extractor (backbone) extracts a convolutional feature map from the input image.
# 2. Region Proposal Network (RPN) predicts candidate object bounding boxes from the feature map.
# 3. RoI pooling crops and resizes regions from the feature map for each proposal.
# 4. Classifier predicts object classes, and bbox regressor predicts refined box offsets.

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, image_shape, in_channels= 2048, mid_channels= 1024, roi_output_size = (3,3)):
        super().__init__()

        # Backbone feature extractor generates the convolutional feature map for the image.
        self.feature_extractor = FeatureExtractor()

        # Region Proposal Network generates candidate object proposals from the feature map.
        self.rpn = RegionProposalNetwork(in_channels, mid_channels, image_shape)

        # RoI pooling layer resizes each proposal region to a fixed output size.
        self.roi_pooling = RoIPooling(roi_output_size)

        # Flattened dimension after RoI pooling: channels * pooled height * pooled width.
        flat_dim = in_channels * roi_output_size[0] * roi_output_size[1]

        # Classifier head predicts class logits for each proposal.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim , 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

        # BBox regressor head predicts bounding box refinements for each proposal.
        self.bbox_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim , 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes * 4)
        )

    def forward(self,x):
        feature_map = self.feature_extractor(x) #B,3, 224,224 → B, 2048 ,7,7
        proposal_boxes,rpn_reg_layer, rpn_cls, anchors, batch_idx = self.rpn(feature_map) 
        rois_with_idx = torch.cat([batch_idx.unsqueeze(1).float(),proposal_boxes], dim=1)
        pooled_features = self.roi_pooling(feature_map, rois_with_idx)
        pooled_features = pooled_features.squeeze(1)
        class_logits = self.classifier(pooled_features)
        bbox_deltas = self.bbox_regressor(pooled_features)

        return class_logits, bbox_deltas,proposal_boxes,rpn_reg_layer, rpn_cls, anchors, batch_idx

