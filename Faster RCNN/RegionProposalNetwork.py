from torch import nn 
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from torchvision.ops import box_iou


class RegionProposalNetwork(nn.Module):
    '''Receive feature map from backbone >> generate binary logit for classifying or objectifying pixel map within 
anchor boxes and delta for moving anchor boxes towards ground truth using conv>> generate anchors that 
distribute and cover every pixels of the image >> big the best unique boxes which doesn't intercept with 
each other too much >> pick the box proposal boxes using non maximum suppresion'''

    def __init__ (self,in_channels, mid_channels,image_shape, num_anchors = 9, stride = 32): 
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1) 
        # create classification head that predict each anchor to be either background or foreground
        self.cls_layer = nn.Conv2d(mid_channels, num_anchors*2, kernel_size=1)
       # regression head — predicts 4 deltas (dx, dy, dw, dh) per anchor to shift it toward GT
        self.reg_layer = nn.Conv2d(mid_channels, num_anchors*4, kernel_size=1)
        self.stride = stride 
        self.image_shape = image_shape


    def anchor_generator(self, feature_map, anchor_sizes= [128, 256, 512], ratios = [0.5,1,2]):
        '''generates 9 anchors (3 sizes x 3 ratios) at every feature map cell
        centre point is (i+0.5)*stride to align with image pixel space'''
        device = feature_map.device             
        img_h, img_w = self.image_shape            
        feature_map_height, feature_map_width = feature_map.shape[2], feature_map.shape[3] # feature size is used as starting point since the size and ratio will cover the whole image at the max lenght of the feature map's width and height
        
        shift_x = torch.arange(feature_map_width, dtype= torch.float32, device=device) 
        shift_y= torch.arange(feature_map_height, dtype= torch.float32, device= device)

        cy, cx = torch.meshgrid(shift_y,shift_x,indexing='ij')
        cy = (cy.reshape(-1) + 0.5) * self.stride
        cx = (cx.reshape(-1) + 0.5) * self.stride 

        sizes = torch.tensor(anchor_sizes, dtype = torch.float32, device= device)
        ratio = torch.tensor(ratios,dtype= torch.float32, device= device )
        
        wx = (sizes[:,None] * ratio[None].sqrt()).reshape(-1)
        hy = (sizes[:,None] / ratio[None].sqrt()).reshape(-1)

        x1 = (cx[:,None] - wx[None]/2).clamp(0,img_w)
        y1 = (cy[:,None] - hy[None]/2).clamp(0,img_h)
        x2 = (cx[:,None] + wx[None]/2).clamp(0,img_w)
        y2 = (cy[:,None] + hy[None]/2).clamp(0,img_h)
        
        anchors = torch.stack([x1,y1,x2,y2], dim=2)
        return anchors.reshape(-1,4)
    

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
        '''NMS - non maximum suppresion returns  the best non overlapping anchor boxes using their 
        objectiveness (the classfication result from the classification head) as ranking criterion'''
        final_anchor, final_rpn_cls, final_rpn_reg_delta, final_proposal, batch_index = [],[],[],[], []
        for i in range(proposal_boxes.shape[0]):
            current_proposal_box= proposal_boxes[i]
            num_best_box = min(2000,current_proposal_box.shape[0])
            score = torch.softmax(rpn_cls[i], dim=1)[:,1]
            # arrange number of best box score from high to low i.e box with highest objectiveness >> high cls logit
            topkscore, indices = score.topk(num_best_box)
            roi = current_proposal_box[indices]
            proposal_score = score[indices]
           # NMS: suppress any box that overlaps a higher-scoring box by more than 0.7 IoU 
            keep_indices = nms(roi, proposal_score, iou_threshold= 0.7)
            final_idx = indices[keep_indices[:300]]
            k = final_idx.shape[0]
            final_proposal.append(current_proposal_box[final_idx])
            final_rpn_reg_delta.append(rpn_reg_delta[i][final_idx])
            final_rpn_cls.append(rpn_cls[i][final_idx])
            final_anchor.append(anchor[i][final_idx])
            batch_index.append(torch.full((k,),i, dtype= torch.long, device=anchor.device))
        return torch.cat(final_anchor, dim= 0),torch.cat(final_proposal, dim =0),torch.cat(final_rpn_cls, dim=0),torch.cat(final_rpn_reg_delta,dim =0), torch.cat(batch_index,dim=0)
    

    def forward(self,x):
        feat = torch.relu(self.conv(x)) # B,2048,7,7 → B,Mi,7,7     Mi == mid_channel
        cls_logits = self.cls_layer(feat) # B,Mi,7,7 → B,num_anchors * 2,7,7
        reg_layer = self.reg_layer(feat) # B,Mi,7,7 → B,num_anchors * 4,7,7
        anchors = self.anchor_generator(x).unsqueeze(dim=0).expand(x.shape[0],-1,-1) # B,441,4
        rpn_cls = cls_logits.permute(0,2,3,1).reshape(x.shape[0],-1,2) # B,441,2
        rpn_reg_delta = reg_layer.permute(0,2,3,1).reshape(x.shape[0],-1,4) # B, 441,4
        proposal_boxes = self.apply_delta_to_anchor(rpn_reg_delta, anchors) #B,441,4
        final_anchor,final_proposal,final_rpn_cls, final_rpn_reg, batch_idx = self.apply_nms_to_images(anchors,rpn_cls,rpn_reg_delta,proposal_boxes) # B, 300,4
        return final_proposal,final_rpn_reg, final_rpn_cls, final_anchor, batch_idx