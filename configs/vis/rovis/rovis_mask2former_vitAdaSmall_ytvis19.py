_base_ = ['./rovis_mask2former_vitAdaTiny_ytvis19.py']

path_model = 'checkpoints/dense_vitAda_mask2former_coco2017/vitAda_small/vitAda_small_coco2017_dense_iter_368750.pth'
work_dir = 'output'

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
            checkpoint=path_model
        ),
    ),
)
