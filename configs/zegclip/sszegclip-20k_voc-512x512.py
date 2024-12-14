_base_ = [
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
img_size = 512
crop_size = (img_size, img_size)
out_indices = [3, 5, 7, 11]
in_channels = 512
base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
novel_class = [15, 16, 17, 18, 19]
both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
num_classes = len(base_class)
pretrained = r"/mnt/ssd/home/jcheng/EfficientNet/YOLOXX/checkpoints/ViT-B-16.pt"
# model_cfg

model = dict(
    type='OSZegCLIP',
    base_class=base_class,
    novel_class=novel_class,
    both_class=both_class,
    pretrained = pretrained,
    backbone=dict(
        type='CLIP_surgery_VisionTransformer',
        input_resolution=512,
        out_indices = out_indices,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        attn_surgery=False,
        vpt_mode=True,
        num_tokens=20, 
        total_d_layer=11,
        query_decoder_in_dim=1024,
        query_decoder_out_dim=100),
    text_encoder= dict(
        type='DynamicPromptCLIPTextEncoder',
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        embed_dim=512,
        tpt_mode = False,
        n_ctx=4,
        prompts_depth=9
    ),
    decode_head=dict(
        type='ATMSingleHeadSeg',
        img_size=img_size,
        in_channels = in_channels,
        channels=in_channels,
        num_classes =num_classes,
        embed_dims=in_channels,
        base_class=base_class,
        both_class=both_class,
        use_stages=1,
        num_layers=3,
        num_heads=8,
        use_proj=True,
        loss_decode=dict(
            type='SegLossPlus', num_classes=num_classes, dec_layers=3, 
            mask_weight=100.0, #20.0
            dice_weight=1.0,
            loss_weight=1.0),
       
    ),
    prompt_learner = dict(
        type='MultiGranularityPromptLearner',
        classnames=class_names,
        seen_idx=base_class,
        all_idx=both_class,
        N_CTX=8,
        n_layers=4,
        input_dim=512,
        prompt_embedding_dim=512,
        content_dim=3,
        patch_size = 32
    ),
    agnostic_mask_generater = None,
    exclude_key=['prompt'],
    load_text_embedding='configs/_base_/datasets/text_embedding/voc12_single.npy',
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(img_size, img_size), stride=(426, 426)), 
    

)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.00002, weight_decay=0.01, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=10.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'prompt_learner':dict(lr_mult=10.0),
                                        'norm': dict(decay_mult=0.),
                                        'ln': dict(decay_mult=0.),
                                        'head': dict(lr_mult=10.),
                                        }))

data = dict(samples_per_gpu=4,
            workers_per_gpu=4,)