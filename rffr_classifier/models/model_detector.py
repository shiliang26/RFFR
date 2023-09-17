from timm.models.vision_transformer import VisionTransformer
from models.model_mae import mae_vit_base_patch16
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from configs.config import config
import torch.nn.functional as F

# Configuration for ViT Base with input 224
cfg = { 'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 
        'mlp_ratio': 4, 'qkv_bias': True, 'norm_layer': partial(nn.LayerNorm, eps=1e-6)}
weights = config.pretrained_weights

class BlockClassifier(VisionTransformer):
    def __init__(self, **kwargs):
        super(BlockClassifier, self).__init__(**kwargs)
        self.load_state_dict(torch.load(weights))
    
    def forward(self, x, patch_id=None):
        D = self.embed_dim
        B = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if patch_id is not None:
            # partial: [Batchsize, L_keep]
            # indexes in partial are the ones to keep.
            patch_id += 1    # This is to skip over the class token.
            patch_id = patch_id.cuda().type(torch.int64) 
            x_masked = torch.gather(x, dim=1, index=patch_id.unsqueeze(-1).repeat(1, 1, D))
            x = x_masked
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        return x[:, 0]

class ResidualGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        mae = mae_vit_base_patch16().cuda()
        mae.load_state_dict(torch.load(config.mae_path)["state_dict"], strict=True)
        self.inpainter = mae
        # This is the index offsets of a 3 * 3 region on a 14 * 14 grid.
        self.magic = torch.Tensor([0, 1, 2, 14, 15, 16, 28, 29, 30])
        self.full_patches = self.block_to_patch(torch.arange(16).unsqueeze(0))

    def block_to_patch(self, idx, test=False):

        bs, blk = idx.shape
        if test:
            return self.full_patches.repeat(bs, 1)

        w, h = 3 * (torch.div(idx, 4, rounding_mode='floor')) + 1, 3 * (idx % 4) + 1
        idx = w * 14 + h

        idx = idx.repeat(1, 9).reshape(bs, -1, blk).transpose(1, 2)
        idx = idx + self.magic
        idx = idx.reshape(bs, -1)
        # [batch_size, num_samples * 9]
        return idx

    def forward(self, rgb_01, test=False):

        if test:
            block_id = None     
        else:
            blocks = []
            for i in range(16):
                if np.random.rand(1) < 0.25:
                    blocks.append(i)
            if len(blocks) == 0:
                block_id = None
            else:
                block_id = torch.LongTensor(blocks).unsqueeze(0).repeat(rgb_01.shape[0], 1)
        
        with torch.no_grad():
            res, block_id = self.inpainter.patch_by_patch_DIFF(rgb_01, block_id = block_id, test=test)
        patch_id = self.block_to_patch(block_id, test=test)

        return res, patch_id

class DeepfakeDetector(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.backbone_1 = BlockClassifier(**cfg)
        self.backbone_2 = BlockClassifier(**cfg)
        self.classifier = nn.Linear(cfg['embed_dim'] * 2, 2)
    
    def forward(self, res, rgb_norm, patch_id):

        feature_1 = self.backbone_1(rgb_norm, patch_id=patch_id)
        feature_2 = self.backbone_2(res, patch_id=patch_id)
        feature = torch.cat([feature_1, feature_2], dim=1)
        output = self.classifier(feature)
        
        return feature, output

class RFFRL(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.rg = ResidualGenerator()
        self.dd = DeepfakeDetector()
    
    def forward(self, rgb_01, rgb_norm, test=False):

        res, patch_id = self.rg(rgb_01, test=test)
        feature, output = self.dd(res, rgb_norm, patch_id)
        
        return feature, output


if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    
    dummy = torch.randn(2, 3, 224, 224).cuda()
    dummy_01 = torch.rand(2, 3, 224, 224).cuda()
    net = RFFRL().cuda()

    result = net(dummy_01, dummy, test=False)
    print(result[0].shape)