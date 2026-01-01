import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: 2 channels (water mask + fat mask)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1)   # final mask logits

    def forward(self, mask_w_logits, mask_f_logits):
        """
        mask_w, mask_f: [B, 256, 256]
        Returns: fused mask logits [B, 1, 256, 256]
        """
        x = torch.stack([mask_w_logits, mask_f_logits], dim=1)    # -> [B, 2, 256, 256]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        out = self.conv3(x)

        return out   # logits
    
# Helper Functions that allows random parameter initialization of the fusion head
    
def init_fusionNet_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MultiModalSAM(nn.Module):
    def __init__(self, sam_water, sam_fat, fusion_net):
        super().__init__()
        self.sam_water = sam_water   # SAM model 1
        self.sam_fat = sam_fat       # SAM model 2
        self.fusion = fusion_net     # FusionNet
        
    def forward(self, fat_pixel_vals, water_pixel_vals, input_boxes):
        """
        img_fat, img_water = [B, 3, H, W] tensors (SAM input format)
        
        Expects each SAM model to output logits:
            [B, 256, 256]
        """
        
        # 1. Run each SAM independently
        mask_f_logits = self.sam_fat(
            pixel_values=fat_pixel_vals,
            input_boxes=input_boxes,
            multimask_output=False).pred_masks.squeeze(1).squeeze(1)          # [B, 256, 256]
                
        mask_w_logits = self.sam_water(
            pixel_values=water_pixel_vals,
            input_boxes=input_boxes,
            multimask_output=False).pred_masks.squeeze(1).squeeze(1)     # [B, 256, 256]

        # 2. Fuse: returns [B, 1, 256, 256]
        fused_logits = self.fusion(mask_w_logits, mask_f_logits)

        return fused_logits, mask_w_logits, mask_f_logits
