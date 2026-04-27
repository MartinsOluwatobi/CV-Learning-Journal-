from torch import nn 
import torch 
import torch.functional as F
class RoIPooling(nn.Module):
    def __init__ (self, output_size, spatial_scale = 0.0625):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_map, rois ):
        pooled_features = []
        h, w = feature_map.shape[2], feature_map.shape[3]
        for roi in rois:
            x1, y1, x2, y2 = roi
            x1 = int(x1 * self.spatial_scale)
            y1 = int(y1 * self.spatial_scale)
            x2 = int(x2 * self.spatial_scale)
            y2 = int(y2 * self.spatial_scale)

            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))

            roi_feature = feature_map[:, :, y1:y2, x1:x2]
            pooled_feature = F.adaptive_max_pool2d(roi_feature, self.output_size)
            pooled_features.append(pooled_feature)
        return torch.stack(pooled_features)