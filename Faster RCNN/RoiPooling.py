from torch import nn 
import torch 
import torch.nn.functional as F
class RoIPooling(nn.Module):
    """Region of Interest Pooling module.

    Converts each RoI proposal into a fixed-size feature tensor by
    cropping a region from the feature map and applying adaptive max pooling.
    """
    def __init__ (self, output_size, spatial_scale = 0.03125):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, feature_map, rois ):
        pooled_features = []
        h, w = feature_map.shape[2], feature_map.shape[3]
        for roi in rois:
            batch, x1, y1, x2, y2 = roi
            batch_idx = int(batch)
            x1 = int(x1 * self.spatial_scale)
            y1 = int(y1 * self.spatial_scale)
            x2 = int(x2 * self.spatial_scale)
            y2 = int(y2 * self.spatial_scale)

            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))

            min_h, min_w = self.output_size
            if y2 == y1:
                y2 = min(y1 + min_h, h)
            if x2 == x1:
                x2 = min(x1 + min_w, w)

            roi_feature = feature_map[batch_idx:batch_idx+1, :, y1:y2, x1:x2]
            if roi_feature.shape[2] == 0 or roi_feature.shape[3] == 0:
                pooled_feature = torch.zeros(1, feature_map.shape[1], self.output_size[0], self.output_size[1], device=feature_map.device)
            else:
                pooled_feature = F.adaptive_max_pool2d(roi_feature, self.output_size)
            pooled_features.append(pooled_feature)
            
        return torch.cat(pooled_features,dim=0)