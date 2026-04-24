_base_ = ['./rovis_mask2former_r50_ytvis19.py']

custom_imports = dict(imports=['custom_mmtrack', 'custom_mmdet'])

path_model = 'checkpoints/dense_vitAda_mask2former_coco2017/vitAda_tiny/vitAda_tiny_coco2017_dense_iter_368750.pth'
work_dir = 'output'

model = dict(
    detector=dict(
        backbone=dict(
            _delete_=True,
            type='ViTAdapter',
            patch_size=16,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4,
            drop_path_rate=0.1,
            layer_scale=False,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=6,
            cffn_ratio=0.25,
            deform_ratio=1.0,
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
            window_attn=[False] * 12,
            window_size=[None] * 12,
            pretrained=None
        ),
        panoptic_head=dict(
            in_channels=[192, 192, 192, 192],  # pass to pixel_decoder inside
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=path_model),
    ),
)

custom_hooks=[
    dict(type='EmptyCacheHook'),
]

val_evaluator = [
    dict(
        _scope_= 'mmtrack',
        type='YouTubeVISMetric',
        metric='youtube_vis_ap',
        outfile_prefix=work_dir + '/results_for_youtube_vis_metric',
        format_only=False),
    dict(
        _scope_= 'mmdet',
        type='CocoMetric_CovertVideo2Image',
        ann_file=_base_.data_root + 'annotations/youtube_vis_2019_valid.json',
        metric=['bbox', 'segm'],
        outfile_prefix=work_dir + '/results_for_coco_metric',
        format_only=False,
        is_video_dataset=True)
    ]
test_evaluator = val_evaluator


# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001 / 4,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            '_delete_': True,
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))


total_epochs = 6
param_scheduler = dict(
    type='mmdet.MultiStepLR',
    begin=0,
    end=total_epochs,
    by_epoch=True,
    milestones=[4],
    gamma=0.1)

val_interval = 1
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=2,
        interval=1))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
