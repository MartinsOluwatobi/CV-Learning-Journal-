from torch import nn 
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from torchvision.ops import box_iou

class RegionProposalNetwork(nn.Module):
    def __init__ (self,in_channels, mid_channels,image_shape, num_anchors = 9, stride = 32): # assume mid_channel = 1024
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1) 
        self.cls_layer = nn.Conv2d(mid_channels, num_anchors*2, kernel_size=1)
        self.reg_layer = nn.Conv2d(mid_channels, num_anchors*4, kernel_size=1)
        self.stride = stride 
        self.image_shape = image_shape


    def anchor_generator(self, feature_map, anchor_sizes= [32, 64, 128], ratios = [0.5,1,2]):
        feature_map_height, feature_map_width = feature_map.shape[2], feature_map.shape[3]
        anchors = []
        ratios = torch.tensor(ratios, dtype=torch.float32, device= feature_map.device)
        anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32,device= feature_map.device)
        img_h, img_w = self.image_shape
        for y in range(feature_map_height):
            for x in range(feature_map_width):
                ix = (x + 0.5) * self.stride
                iy = (y + 0.5) * self.stride
                for size in anchor_sizes:
                    for ratio in ratios:
                        w = size * torch.sqrt(ratio)
                        h = size / torch.sqrt(ratio)
                        x1 =torch.clamp(ix - w/2, min = 0, max = img_w)
                        y1 = torch.clamp(iy- h/2, min = 0, max = img_h)
                        x2 = torch.clamp(ix + w/2, min = 0, max = img_w)
                        y2 = torch.clamp(iy + h/2, min = 0, max = img_h)
                        anchors.append(torch.stack([x1,y1,x2,y2]))
        return torch.stack(anchors) 
    
    def apply_delta_to_anchor (self, delta, anchor):
        widths  = anchor[:, :, 2] - anchor[:, :, 0]
        heights = anchor[:, :, 3] - anchor[:, :, 1]
        ctr_x   = anchor[:, :, 0] + 0.5 * widths
        ctr_y   = anchor[:, :, 1] + 0.5 * heights
        dx, dy  = delta[:, :, 0], delta[:, :, 1]
        dw, dh  = delta[:, :, 2], delta[:, :, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(torch.clamp(dw, max=4.0)) * widths
        pred_h = torch.exp(torch.clamp(dh, max=4.0)) * heights

        x1 = pred_ctr_x - (0.5 * pred_w)
        y1 = pred_ctr_y - (0.5 * pred_h)
        x2 = pred_ctr_x + (0.5 * pred_w)
        y2 = pred_ctr_y + (0.5 * pred_h)

        return torch.stack([x1,y1,x2,y2], dim = 2)
    
    def apply_nms_to_images(self,anchor,rpn_cls,rpn_reg_delta,proposal_boxes):
        final_anchor, final_rpn_cls, final_rpn_reg_delta, final_proposal, batch_index = [],[],[],[], []
        for i in range(proposal_boxes.shape[0]):
            current_proposal_box= proposal_boxes[i]
            topk = min(2000,current_proposal_box.shape[0])
            score = torch.softmax(rpn_cls[i], dim=1)[:,1]
            _, indices = score.topk(topk)
            roi = current_proposal_box[indices]
            proposal_score = score[indices]
            keep_indices = nms(roi, proposal_score, iou_threshold= 0.7)
            final_idx = indices[keep_indices[:300]]
            k = final_idx.shape[0]
            final_proposal.append(current_proposal_box[final_idx])
            final_rpn_reg_delta.append(rpn_reg_delta[i][final_idx])
            final_rpn_cls.append(rpn_cls[i][final_idx])
            final_anchor.append(anchor[i][final_idx])
            batch_index.append(torch.full((k,),i, dtype= torch.long))
        return torch.cat(final_anchor, dim= 0),torch.cat(final_proposal, dim =0),torch.cat(final_rpn_cls, dim=0),torch.cat(final_rpn_reg_delta,dim =0), torch.cat(batch_index,dim=0)
    

    def forward(self,x):
        feat = torch.relu(self.conv(x)) # B,1024,7,7
        cls_logits = self.cls_layer(feat) # B, 18, 7,7 
        reg_layer = self.reg_layer(feat) # B, 36, 7,7

        anchors = self.anchor_generator(x).unsqueeze(dim=0).expand(x.shape[0],-1,-1) # B,441,4
        rpn_cls = cls_logits.permute(0,2,3,1).reshape(x.shape[0],-1,2) # B,441,2
        rpn_reg_delta = reg_layer.permute(0,2,3,1).reshape(x.shape[0],-1,4) # B, 441,4
        proposal_boxes = self.apply_delta_to_anchor(rpn_reg_delta, anchors) #B,441,4
        final_anchor,final_proposal,final_rpn_cls, final_rpn_reg, batch_idx = self.apply_nms_to_images(anchors,rpn_cls,rpn_reg_delta,proposal_boxes)
        return final_proposal,final_rpn_reg, final_rpn_cls, final_anchor, batch_idx