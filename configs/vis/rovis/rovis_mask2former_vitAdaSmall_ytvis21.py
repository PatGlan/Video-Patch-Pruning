_base_ = ['./rovis_mask2former_vitAdaTiny_tracker.py']

#path_model = 'PATH_MODEL_PRETRAINED_ON_COCO2017'
path_model = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'

model = dict(
    detector=dict(
        backbone=dict(
            _delete_=True,
            type='ViTAdapter',
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            drop_path_rate=0.2,
            layer_scale=False,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=6,
            cffn_ratio=0.25,
            deform_ratio=1.0,
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
            window_attn=[False] * 12,
            window_size=[None] * 12,
            pretrained=None,
        ),
        panoptic_head=dict(
            in_channels=[384, 384, 384, 384],  # pass to pixel_decoder inside
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=path_model.replace('.pth', '_evo.pth')),
    ),
)

_base_.custom_hooks[0]=dict(type='UpdateSvitCkptHook', model_path=path_model, priority='HIGHEST')