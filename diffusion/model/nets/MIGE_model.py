# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import os
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.t5 import T5Embedder
from diffusion.model.nets.MIGE_blocks import t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, LabelEmbedder, FinalLayer
from diffusion.utils.logger import get_root_logger
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from transformers import T5TokenizerFast
from lavis.models.blip2_models.modeling_t5 import T5Config, T5EncoderModel
from lavis.models.eva_vit import VisionTransformer
from functools import partial
from lavis.models.blip2_models.blip2 import LayerNorm
from transformers import Dinov2Model, Dinov2Config
from diffusion.model.nets.FFM_module import FFM

def create_print_grad_hook(tensor_name):
    def print_grad(grad):
        #print(f"Gradient for {tensor_name}: {grad}")
        pass
    return print_grad
    
class MIGEBlock(nn.Module):
    """
    A MIGE block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


#############################################################################
#                                 Core MIGE Model                                #
#################################################################################
@MODELS.register_module()
class MIGE(Blip2Base):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, 
                 vit_model = "eva_clip_g", vit_path = None,blip2 = None, img_size=224,max_len = 256, drop_path_rate=0, use_grad_checkpoint=False, vit_precision="float32",freeze_vit=False, num_query_token=32, cross_attention_freq=2,embed_dim=256, max_txt_len=32,t5_model="output/pretrained_models/t5-v1_1-xxl",
                 input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=True, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=4096, lewei_scale=1.0, config=None, model_max_length=256, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        ###### BLIP2
        self.max_len = max_len

        self.visual_encoder =  VisionTransformer(
            img_size=img_size, #224
            patch_size=14,
            use_mean_pooling=False,
            embed_dim=1408,
            depth=39,
            num_heads=1408//88,
            mlp_ratio=4.3637,
            qkv_bias=True,
            drop_path_rate=drop_path_rate, #0
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_checkpoint = False,
        )
        self.ln_vision = LayerNorm(self.visual_encoder.num_features)
        print('finish init vit without loading param')

        self.Qformer, self.query_tokens = self.init_Qformer( #在blip2.py里进行的加载，未加载参数
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer: # 12层
            layer.output = None
            layer.intermediate = None
            
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)

        self.t5_model = T5EncoderModel(config=t5_config)
        print('finish init T5 without loading param')
        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        ) # Linear(in_features=1536, out_features=4096, bias=True)
        self.max_txt_len = max_txt_len #32
        self._lemmatizer = None

        self.FFM = FFM()
        
        #####
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels #4
        self.out_channels = in_channels * 2 if pred_sigma else in_channels #8
        self.patch_size = patch_size #2
        self.num_heads = num_heads #16
        self.lewei_scale = lewei_scale, #1

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels*2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size) #linear(256,1152),silu,linear(1152,1152)
        num_patches = self.x_embedder.num_patches #256像素的就是patch个数16*16=256，512像素的就是32*32=1024
        self.base_size = input_size // self.patch_size # 一边长多少个patch。512的就32，256的就16
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size)) #[1,256,1152]

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential( #alpha1,2 beta1,2 gamma1,2
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)
        drop_path = [ x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            MIGEBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(input_size // patch_size, input_size // patch_size),
                          window_size=window_size if i in window_block_indexes else 0,
                          use_rel_pos=use_rel_pos if i in window_block_indexes else False)
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()
        self.FFM.apply(self.FFM._init_weights)


        if config:
            logger = get_root_logger(os.path.join(config.work_dir, 'train.log')) 
            logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        else:
            print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')

    def get_context_emb(self, prompts, img_lists):
        
        # 为每个段落和图像创建批处理的嵌入列表
        batch_img_embs = []
        batch_attn = []
        
        for prompt, img_list in zip(prompts, img_lists):
            prompt_segs = prompt.split('<imagehere>')
            seg_tokens = [
                self.t5_tokenizer(seg, return_tensors="pt", add_special_tokens=False).to(self.device).input_ids
                for seg in prompt_segs] 
            if seg_tokens:  
                seg_tokens[-1] = torch.cat([
                    seg_tokens[-1],
                    torch.tensor([[self.t5_tokenizer.eos_token_id]], device=self.device)
                ], dim=-1)
            seg_embs = [self.t5_model.encoder.embed_tokens(seg_t.long()) for seg_t in seg_tokens]
            mixed_embs = [emb for pair in zip(seg_embs[:-1], [img.unsqueeze(0) for img in img_list]) for emb in pair] + [seg_embs[-1]]
            mixed_embs = torch.cat(mixed_embs, dim=1) #[1,len,4096]
            
            
            padding_length = self.max_len - mixed_embs.size(1)
            if (padding_length == 255):
                print('prompt:',prompt,len(img_list),' emb:', mixed_embs)

            if padding_length > 0:
                padding_emb = self.t5_model.encoder.embed_tokens(torch.tensor([0], device=self.device))
                batch_size, _, embedding_dim = mixed_embs.shape
                padding = padding_emb.repeat(batch_size, padding_length, 1)
                inputs_embeds_padded = torch.cat([mixed_embs, padding], dim=1)
                attention_mask = torch.cat([
                    torch.ones((mixed_embs.size(0), mixed_embs.size(1)), dtype=torch.long, device=img_list.device),
                    torch.zeros((padding.size(0), padding_length), dtype=torch.long, device=img_list.device)
                ], dim=1)
            else:
                inputs_embeds_padded = mixed_embs[:,:self.max_len]
                attention_mask = torch.ones((mixed_embs.size(0), self.max_len), dtype=torch.long, device=img_list.device)
            batch_img_embs.append(inputs_embeds_padded)
            batch_attn.append(attention_mask)

        batch_img_embs = torch.cat(batch_img_embs,dim=0)
        batch_attn = torch.cat(batch_attn,dim=0)

        
        return batch_img_embs, batch_attn

    
    def forward(self, x, timestep, y, source = None, ref_clip =None, ref_vae = None, mask = None, state = None, data_info=None, **kwargs):
        # y: multimodal instruction
        # ref : reference image
        """
        Forward pass of MIGE.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 256, C) condition
        """
        ###### Q-Former
        ## ref : [bs, 4, 3, 224, 224] bf16
        images = ref_clip.view(-1, 3, 224, 224)  # Reshape to [bs*4, 3, 224, 224] float32
        vae_images = ref_vae.view(-1, 4, 64, 64).to(self.dtype)
        zeros = torch.zeros_like(vae_images)
        vae_images_extended = torch.cat([vae_images, zeros], dim=1)
        vae_features = self.x_embedder(vae_images_extended.to(self.device)) #[bs*4,1024,1152]

        # mask，identify all the non-zero images
        non_zero_mask = (images != 0).any(dim=3).any(dim=2).any(dim=1) # [bs*4]

        non_zero_images = images[non_zero_mask]
        
        if len(non_zero_images)!=0 :

            with self.maybe_autocast(): 
                image_embeds = self.ln_vision(self.visual_encoder(images)).to(torch.bfloat16)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(x.device)  # [bs*4, 257]

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # [bs*4, 32, 768]
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,#bf16
                encoder_hidden_states=image_embeds,  # [bs*4, 257, 1408] bf16
                encoder_attention_mask=image_atts,  # [bs*4, 257]
                return_dict=True
            )#[bs*4,32,768]

            t5_proj_input = self.FFM(clip = query_output.last_hidden_state, vae = vae_features) #[bs*4,32,768]
            
            t5_proj_emb = self.t5_proj(t5_proj_input) #[bs*4,32,4096] bf16
            t5_proj_emb = t5_proj_emb.view(-1, 4, *t5_proj_emb.shape[1:]) # [bs,4,32,4096]

        else:
            t5_proj_emb = torch.zeros([ref_clip.shape[0],4,32,4096],dtype=ref_clip.dtype).to(ref_clip.device)

        # mix text and image
        batch_mixed_embs, batch_attn = self.get_context_emb(y, t5_proj_emb)  # [bs, 256, 4096]float32 [bs,256]int64
        mask = batch_attn

        text_encoder_embs = self.t5_model( #[bs,256,4096]  bf16
            inputs_embeds = batch_mixed_embs,
            attention_mask = mask,
        )['last_hidden_state']
        
        y = text_encoder_embs[:, None].to(self.dtype) #[bs,1,256,4096]

        # for DPM
        null_y = self.y_embedder.y_embedding[None].repeat(text_encoder_embs.shape[0], 1, 1)[:, None] #[bs,1,256,4096]
        
        if state == 'subject':
            y = torch.cat([null_y, y]) #[bs*2,1,256,4096]
            source = torch.cat([source,source])
        elif state == 'edit':
            y = torch.cat([null_y, null_y, y])
            null_s = torch.zeros(source.shape[0],4,64,64)
            source = torch.cat([null_s,source,source])
        #####
        x = x.to(self.dtype).to(self.device)
        source = source.to(self.dtype).to(self.device)

        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size #16,16
        
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D) [bs,1152]
        t0 = self.t_block(t) 

        if self.training:
            # 5% drop y or source, 5% drop all
            batch_size = y.shape[0]
            drop_prob = torch.rand(batch_size).to(self.device)

            y_mask = (drop_prob < 0.05) | (drop_prob >= 0.10) & (drop_prob < 0.15) 
            source_mask = (drop_prob >= 0.05) & (drop_prob < 0.10) | (drop_prob >= 0.10) & (drop_prob < 0.15)

            y_replacement = self.y_embedder.y_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [bs, 1, 256, 4096]
            source_replacement = torch.zeros_like(source.to(self.dtype))

            if y_mask.any():
                y[y_mask] = y_replacement[y_mask]
            if source_mask.any():
                source[source_mask] = source_replacement[source_mask]
            
        x = torch.cat([x,source],dim=1) #[bs,8,64,64]
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2 [2,64*64/4,1152]
        y = self.y_embedder(y, self.training)  # (N, 1, L, D) #[bs,1,max_len(256),4096->1152]

        if mask is not None: 
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1) 
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels) [2,256,32]
        x = self.unpatchify(x)  # (N, out_channels, H, W) #[2,8,32,32]

        return x

    def forward_with_dpmsolver(self, x, timestep, cond, state = None, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        
        if type(cond[0]) == dict: #multimodal
            y = []
            ref_clip = []
            ref_vae = []
            source = []
            for prompt in cond:
                y.append(prompt['prompt'])
                ref_clip.append(prompt['ref_clip'])
                ref_vae.append(prompt['ref_vae'])
                source.append(prompt['source'])
            ref_clip = torch.stack(ref_clip, dim=0)
            ref_vae = torch.stack(ref_vae, dim=0)
            source = torch.stack(source,dim=0)
            model_out = self.forward(x, timestep, y, source = source, ref_clip=ref_clip, ref_vae=ref_vae, mask=mask, state = state) #没有mask
        else: 
            model_out = self.forward(x, timestep, cond, mask) 
        return model_out.chunk(2, dim=1)[0] #[bs*2,4,64,64]

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in MIGE blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    #grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
    #grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
    grid_h = np.arange(grid_size[0]) / (grid_size[0]/base_size) / lewei_scale
    grid_w = np.arange(grid_size[1]) / (grid_size[1]/base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    #omega = np.arange(embed_dim // 2)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


#################################################################################
#                                   MIGE Configs                                  #
#################################################################################
@MODELS.register_module()
def MIGE_XL_2(**kwargs):
    return MIGE(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
