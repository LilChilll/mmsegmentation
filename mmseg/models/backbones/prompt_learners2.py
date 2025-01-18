from audioop import tomono
import copy
import math
from os import TMP_MAX
import os.path as osp
from collections import OrderedDict
from mmseg.models.builder import BACKBONES
import torch
import torch.nn as nn
# from timm.models.layers import drop, drop_path, trunc_normal_
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import sys
# sys.path.append("..")
# sys.path.append(".")
# from .utils import *
# from clip import clip
from ..utils.tokenize import tokenize
# from clip import tokenize
from torch.nn import Dropout
templates = [
    'a photo of a {}.',
    'a photo of a small {}.',
    'a photo of a medium {}.',
    'a photo of a large {}.',
    'This is a photo of a {}.',
    'This is a photo of a small {}.',
    'This is a photo of a medium {}.',
    'This is a photo of a large {}.',
    'a {} in the scene.',
    'a photo of a {} in the scene.',
    'There is a {} in the scene.',
    'There is the {} in the scene.',
    'This is a {} in the scene.',
    'This is the {} in the scene.',
    'This is one {} in the scene.',
    ]

@BACKBONES.register_module()
class MultiGranularityPromptLearner(nn.Module):
    """
        多粒度提示词生成器，包含风格投影器、内容投影器以及一个融合模块
        通过来自Image encoder的多层局部视觉特征，提取风格特征和内容特征，然后融合生成提示词

    """
    def __init__(self,
                 classnames,
                 seen_idx=[],
                 all_idx=[],
                 input_dim=512,
                 prompt_embedding_dim=512,
                 content_dim=3,
                 patch_size = 32,
                 N_CTX=4,
                 n_layers = 4,
                 CTX_INIT = None,
                 ensemble_prompts= False,
                 training = True
                 ):
        super().__init__()
        self.training=training
        self.input_dim = input_dim
        self.prompt_embedding_dim = prompt_embedding_dim
        self.content_dim = content_dim
        self.patch_size = patch_size
        self.n_layers=n_layers
        
        self.n_ctx = N_CTX
        self.ctx_init = CTX_INIT
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.classnames = [name for i,name in enumerate(classnames) if i in seen_idx]
        self.all_classnames = classnames
        if self.training:
            self.n_cls =len(self.classnames)
        else:
            self.n_cls = len(self.all_classnames)
        self.ensemble_prompts = ensemble_prompts
        


    def init_context(self,clip_model):
        classnames = self.classnames
        input_dim =  self.input_dim
        prompt_embedding_dim=self.prompt_embedding_dim
        content_dim=self.content_dim
        patch_size = self.patch_size
        n_layers = self.n_layers
        dtype = clip_model.dtype
        n_cls = self.n_cls
        n_ctx = self.n_ctx
        ctx_init = self.ctx_init
        classnames = [name.replace("_", " ") for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        if self.ensemble_prompts:
            ensemble_prompt_embeddings = []
            ensemble_prompts = []
            for classname in classnames:
                prompts = [template.format(classname) for template in templates] #format with class
                tokenized_prompts = torch.cat([tokenize(p) for p in prompts]) # c,77
                tokenized_prompts = tokenized_prompts.to(next(clip_model.parameters()).device)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # c,77,512
                ensemble_prompt_embeddings.append(embedding)
                ensemble_prompts.append(tokenized_prompts)
            ensemble_prompt_embeddings = torch.stack(ensemble_prompt_embeddings, dim=0)  # [n_templates, c, 77, 512]
            ensemble_prompts = torch.stack(ensemble_prompts, dim=0)  # [n_templates, c, 77]
            self.register_buffer("token_prefix", ensemble_prompt_embeddings[:,:, :1, :])  # SOS # n_templates,c,1,512
            self.register_buffer("token_suffix", ensemble_prompt_embeddings[:,:, 1 + n_ctx:, :])  # CLS, EOS # n_templates,c,60,512
            self.register_buffer("tokenized_prompts", ensemble_prompts)  # n_templates,c,77
        else:
            if ctx_init:
                ctx_init = ctx_init.replace("_", " ")
                self.n_ctx = len(ctx_init.split(" "))
                prompt_prefix = ctx_init
            else:
                prompt_prefix = " ".join(["X"] * n_ctx)
            self.prompt_prefix=prompt_prefix
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([tokenize(p) for p in prompts]) # c,77
            tokenized_prompts = tokenized_prompts.to(next(clip_model.parameters()).device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # c,77,512
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS # c,1,512
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS #c,60,512
            self.register_buffer("tokenized_prompts", tokenized_prompts)  # c,77
            
        for i in range(n_ctx):
            style_projector = nn.Sequential(
                                    nn.Linear(input_dim*2, 512),  # 注意这里的input_dim应该是原特征维度的两倍
                                    nn.ReLU(),
                                    nn.Linear(512, prompt_embedding_dim)
                                )
            setattr(self, f"style_projector{i}", style_projector)

        self.bottleneck = nn.Sequential(
                nn.Conv2d(input_dim, content_dim, 1),
                nn.Flatten()
        )
        self.content_proj =nn.Sequential(
            nn.Linear(n_layers*content_dim*patch_size*patch_size, 512),
                nn.ReLU(),
                nn.Linear(512, prompt_embedding_dim))
        self.fusion_proj = nn.Sequential(
            nn.Linear(prompt_embedding_dim, 512),
                nn.ReLU(),
                nn.Linear(512, prompt_embedding_dim))  #nn.Parameter(scale * torch.rand(n_cls,n_layers))
        self.fusion_attention = nn.MultiheadAttention(prompt_embedding_dim, 8)


        self.dropout = Dropout(0.1)
        
        

        
        # self.dynamic_weights = nn.Parameter(torch.ones(n_layers, dtype=torch.float32), requires_grad=True)

        self.apply(self._init_weights)

    def generate_prompts_for_testing(self,clip_model):
        prompts = [self.prompt_prefix + " " + name + "." for name in self.all_classnames]
        dtype = clip_model.dtype
        n_ctx = self.n_ctx
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]) # c,77
        tokenized_prompts = tokenized_prompts.to(next(clip_model.parameters()).device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # c,77,512
        new_token_prefix = embedding[:, :1, :]  # SOS # c,1,512
        new_token_suffix = embedding[:, 1 + n_ctx:, :]  # CLS, EOS #c,60,512
        new_tokenized_prompts = tokenized_prompts  # c,77
        new_token_prefix[self.seen_idx] = self.token_prefix
        new_token_suffix[self.seen_idx] = self.token_suffix
        new_tokenized_prompts[self.seen_idx] = self.tokenized_prompts
        self.token_prefix = new_token_prefix
        self.token_suffix = new_token_suffix
        self.tokenized_prompts = new_tokenized_prompts
        print("prompts adjusted.")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            nn.init.trunc_normal_(m.weight,std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def fusion_projector(self,content_feat,text_feat):
        c,_ = text_feat.shape # c,dim
        b,dim = content_feat.shape # b,dim
        content_feat = content_feat.unsqueeze(0).expand(c,-1,-1).permute(1,0,2).contiguous() # b,c,dim
        text_feat = text_feat.unsqueeze(1).expand(-1,b,-1).permute(1,0,2).contiguous() #b,c,dim
        attention_output, _ = self.fusion_attention(content_feat, text_feat, text_feat)

        # 将注意力输出映射到目标维度
        output_features = self.fusion_proj(attention_output.mean(dim=0))  # [n_cls, dim]
        return output_features
    def content_projector(self,im_feature):
        # im_feature : l*c*h*w   
        # error runtime: im_feature:h,w:24,24
        x = self.bottleneck(im_feature)
        x = x.reshape(1,-1)
        x = self.content_proj(x)
        return x
    def prepare_style_img_feature(self,im_feature):
        L, C, n = im_feature.shape
        H = W = int(math.sqrt(n))
        assert H*W == n, "n should be a square number"
        im_feature = im_feature.reshape(L, C, H, W)
        if L > self.n_ctx:
            # 如果层数多于上下文长度，进行层级分组平均
            group_size = L // self.n_ctx
            grouped_features = []
            for start in range(0, L, group_size):
                end = min(start + group_size, L)
                grouped_features.append(im_feature[start:end].mean(dim=0))  # 分组平均
            result = torch.stack(grouped_features, dim=0).to(im_feature.device)  # [N_ctx, C,H,W]

        elif L < self.n_ctx:
            # 如果层数少于上下文长度，进行重复扩展
            replication = im_feature[-1].unsqueeze(0).repeat(self.n_ctx - L, 1, 1, 1)
            result = torch.cat((im_feature, replication), dim=0)  # [N_ctx, C, H,W]

        else:
            # 层数等于上下文长度
            result = im_feature

        return result
    def compute_style_features(self, feature_maps, attention_weights=None):
        """
        改进风格特征的计算逻辑：增强对多尺度统计信息的提取。
        :param feature_maps: 特征图，[L, C, H*W]
        :return: 计算后的风格特征，形状为 [2 * C]
        """
        # 多尺度特征输入，形状为 [L, C, n]
        L, C, H,W = feature_maps.shape
       
        spatial_dims = (1, 2)  # 统计空间维度 [H, W]
        style_features_list = []

        for l in range(L):
            # 对每层特征计算均值和标准差
            current_map = feature_maps[l]
            mean = current_map.mean(dim=spatial_dims)  # [C]
            std = current_map.std(dim=spatial_dims)    # [C]

            # 拼接均值和标准差，形状为 [2 * C]
            current_style = torch.cat((mean, std), dim=0)
            style_projector = getattr(self,f"style_projector{l}")
            current_style = style_projector(current_style)
            style_features_list.append(current_style)

        
        style_features = torch.stack(style_features_list, dim=0)  # [2 * C]

        return style_features  # 输出形状 [2 * C]
    def forward(self, im_features,text_encoder):
        if self.ensemble_prompts:
            return self.forward_ensemble(im_features,text_encoder)
        else:
            return self.forward_single(im_features,text_encoder)
    def forward_single(self, im_features,text_encoder):
        """

        :param im_features:(b,h*w,d)*L
        :return:
        """
        im_features = torch.stack(im_features,dim=0) # L,b,1024,512
        im_features = im_features.permute(1,0,3,2).contiguous() # b,L,512,1024
        # print(im_features.shape)
        bs,l,c,n = im_features.shape
        w = h=int(math.sqrt(n))
        assert w*h == n, "n should be a square number"
        # prompt_prefix = "a photo of a {}."
        # prompts = [prompt_prefix.format(name) for name in self.classnames]
        # tokenized_prompts = torch.cat([tokenize(p) for p in prompts]) # c,77
        # tokenized_prompts = tokenized_prompts.to(next(text_encoder.parameters()).device)
        # with torch.no_grad():
        #     class_embedding = text_encoder.token_embedding(tokenized_prompts).type(text_encoder.dtype) # c,77,512
        #     token_prefix = class_embedding[:, :1, :]  # SOS # c,1,512
        #     token_suffix = class_embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS #c,60,512
        prefix = self.token_prefix.expand(bs,-1,-1,-1) # bs,c,1,512
        suffix = self.token_suffix.expand(bs,-1,-1,-1)  #bs,c,72,512
        # n_templates = self.token_prefix.shape[1]
        content = []
        style_features = []
        content_features = []
        for i,img_feat in enumerate(im_features):
            # img_feat: l,c,h*w
            style_img_feat = self.prepare_style_img_feature(img_feat) 
            style_feat = self.compute_style_features(style_img_feat) # n_ctx,dim
            
            feature_token = style_feat.expand(self.n_cls,-1,-1).to(im_features.device) # c,n_ctx,dim
            style_features.append(feature_token)

            content_token = self.dropout(self.content_projector(img_feat.reshape(l,c,h,w)).to(im_features.device)) # 1,dim
            content_features.append(content_token)

        style_features = torch.stack(style_features,dim=0) # b,c,n_ctx,dim
        # style_features,_ = style_features.max(dim=0) # c,n_ctx,dim
        content_features = torch.stack(content_features,dim=0) # b,dim
        ensemble_prompts = torch.cat([
            prefix,
            style_features,
            suffix
        ],
        dim=2) # bs,n_templates,c,77,512
        text_tokens = []
        for pts_i, ctt_i in zip(ensemble_prompts, content_features):
            # pts_i :(c,77,512)
            # ctt_i :(1,512)
            text_features = text_encoder.forward_dynamic_prompts(pts_i, self.tokenized_prompts)            
            text_features = self.fusion_projector(ctt_i, text_features)
            text_tokens.append(text_features)
        text_embedding = torch.stack(text_tokens, dim=0).to(im_features.device)  # c,512
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # c,512
            
    
        return text_embedding
    def forward_ensemble(self, im_features,text_encoder):
        """

        :param im_features:(b,h*w,d)*L
        :return:
        """
        im_features = torch.stack(im_features,dim=0) # L,b,1024,512
        im_features = im_features.permute(1,0,3,2).contiguous() # b,L,512,1024
        bs,l,c,n = im_features.shape
        w = h=int(math.sqrt(n))
        assert w*h == n, "n should be a square number"
        # print(im_features.shape)

        n_templates = len(templates)
        ensemble_prompt_embeddings = []
        ensemble_prompts = []
        for classname in self.classnames:
            prompts = [template.format(classname) for template in templates] #format with class
            tokenized_prompts = torch.cat([tokenize(p) for p in prompts]) # c,77
            tokenized_prompts = tokenized_prompts.to(next(text_encoder.parameters()).device)
            with torch.no_grad():
                class_embedding = text_encoder.token_embedding(tokenized_prompts).type(text_encoder.dtype) # c,77,512
            ensemble_prompt_embeddings.append(class_embedding)
            ensemble_prompts.append(tokenized_prompts)
        ensemble_prompt_embeddings = torch.stack(ensemble_prompt_embeddings, dim=0)  # [n_templates, c, 77, 512]
        ensemble_prompts = torch.stack(ensemble_prompts, dim=0)  # [n_templates, c, 77]
        token_prefix = class_embedding[:, :1, :]  # SOS # c,1,512
        token_suffix = class_embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS #c,60,512
        tokenized_prompts = ensemble_prompts
        
        prefix = token_prefix.expand(bs,-1,-1,-1,-1) # bs,n_templates,c,1,512
        suffix = token_suffix.expand(bs,-1,-1,-1,-1)  #bs,n_templates,c,72,512
        content = []
        style_features = []
        content_features = []
        for i,img_feat in enumerate(im_features):
            # img_feat: l,c,h*w
            style_img_feat = self.prepare_style_img_feature(img_feat) 
            style_feat = self.compute_style_features(style_img_feat) # n_ctx,dim
            
            feature_token = style_feat.expand(self.n_cls,-1,-1).to(im_features.device) # c,n_ctx,dim
            style_features.append(feature_token)

            content_token = self.dropout(self.content_projector(img_feat.reshape(l,c,h,w)).to(im_features.device)) # 1,dim
            content_features.append(content_token)

        style_features = torch.stack(style_features,dim=0) # b,c,n_ctx,dim
        style_features = style_features.unsqueeze(1).expand(-1,n_templates,-1,-1,-1) # b,n_templates,c,n_ctx,dim
        # style_features,_ = style_features.max(dim=0) # c,n_ctx,dim
        content_features = torch.stack(content_features,dim=0) # b,dim
        ensemble_prompts = torch.cat([
            prefix,
            style_features,
            suffix
        ],
        dim=3) # bs,n_templates,c,77,512
        ensemble_text_embeddings = []
        for enspts_i, ctt_i in zip(ensemble_prompts, content_features):
            # pts_i :(n_templates,c,77,512)
            # ctt_i :(1,512)
            text_tokens = []
            for pts_i,tokenized_prompt in zip(enspts_i,tokenized_prompts):
                # pts_i :(c,77,512)
                text_features = text_encoder.forward_dynamic_prompts(pts_i, tokenized_prompt)            
                text_features = self.fusion_projector(ctt_i, text_features)
                text_tokens.append(text_features)
            text_embedding = torch.stack(text_tokens, dim=0)  # n_templates,c,512
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # n_templates,c,512
            text_embedding = text_embedding.mean(dim=0)
            text_embedding = text_embedding / text_embedding.norm()
            ensemble_text_embeddings.append(text_embedding)
        ensemble_text_embeddings = torch.stack(ensemble_text_embeddings, dim=0).to(im_features.device)  # b,c,512

        # text_embedding = text_encoder.forward_dynamic_prompts(prompts, tokenized_prompts) # c,512
        # text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True) # c,512
        # text_embedding = self.fusion_projector(content_features,text_embedding) # c,512
    
        return ensemble_text_embeddings
    