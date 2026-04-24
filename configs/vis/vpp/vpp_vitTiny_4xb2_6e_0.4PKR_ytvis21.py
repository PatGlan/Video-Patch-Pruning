_base_ = [
    '../rovis/rovis_mask2former_vitAdaTiny_ytvis21.py',
]

custom_imports = dict(imports=['custom_mmtrack', 'custom_mmdet'])

#data_root = '/tmp/data/youtube_vis_2021/'
#work_dir = '../runs/tmp/mmTrack/vpp/dataset=ytvis21_model=rovis/method=ivpp/vppTiny_4xb2-lsj-6e_kr0.5_V66'
#path_model = f'../runs/tmp/{save_dir}/vpp/dataset=ytvis21_model=rovis/method=vit/vitTiny_4xb2-lsj-6e/epoch_6.pth'
#find_unused_parameters = True

path_model = 'checkpoints/dense_vitAda_rovis_ytvis21/tiny/epoch_6.pth'
work_dir = 'output'

#model
num_things_classes = 40
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    detector=dict(
        type='Mask2Former',
        backbone=dict(
            type='SelectiveViTAdapter',
            #select_loc=[3, 6, 9],

            selec_skip_ids=[0],  # ids where the selective module is not been applied
            #selec_ids_use_res=[3, 6, 9],
            #tau_update_layer=10, #constolls the noise of initial mask
            #mask_pooling_size=4, #init-mask is pooled to include background patches close to the object
            #scale_attn_update_layer=1000, #controlls entropy attn-score of update module (realtated to top-K-values)
            #initBlock_keep_ratio=0.7,

            #prev_x_update_attn=False, #train matching module with train dynamically
            #apply_mask_loss_to_layers=[], #default: [0], full alterantive = [0, 1, 2, 3]

            pruning_method = 'ivpp',

            #selective moduels
            selective_module=dict(
                type='SelectiveModule',
                emb_dim=192,
                hidden_dim=192//4,
                tau=10, #controlls gumbel noise
                #compute_pruned_layers=False,
                #pool_p=False,
                #test_sampler=False,
            ),
            select_module_ids=[3, 6, 9],
            selective_module_kr=[0.56, 0.448, 0.358],

            #matching module
            matching_module_idx=1,
            prev_x_update_layer=5,
            matching_module=dict(
                type="Mapping_Selective_Module",
                in_dim=192,
                hid_dim=192//2,
                keep_ratio=0.7,
                tau_matching=10,
                tau_selector=10
            ),

            #loss
            ratio_sm_sparsity_loss=40.,
            ratio_match_module_sparsity_loss=10.,
            #ratio_loss_by_all_layers=False,
            #ratio_min_gt_loss=0.,
            #ratio_loss_dist=0.,
            #ratio_fn_mask_loss=0.,

            #scale_loss_first_frame=1.0,
        ),
        panoptic_head=dict(
                    num_things_classes=num_things_classes,
                    num_stuff_classes=num_stuff_classes,
                    loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])
        ),
        panoptic_fusion_head = dict(
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes
        ),
        init_cfg=dict(_delete_=True),
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=path_model.replace('.pth', '_evo.pth')
    ),
)

custom_hooks=[
    dict(type='ChangeKeyNameHook', model_path=path_model, priority='HIGHEST'),
    dict(type='EmptyCacheHook'),
    dict(type='LogBlockSparsity', save_results=True),
]


#dataset
_base_.train_pipeline[1] = dict(type='PackTrackInputs_MultiRefImgs', ref_prefix='ref', num_key_frames=1, reduce_multiple_ref_imgs=True)
train_dataloader = dict(batch_size=1,
                        dataset=dict(
                            pipeline=_base_.train_pipeline,
                            ref_img_sampler=dict(num_ref_imgs=2)
                        )
                    )
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader


'''
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
        ann_file=data_root + 'annotations/youtube_vis_2021_valid.json',
        metric=['bbox', 'segm'],
        outfile_prefix=work_dir + '/results_for_coco_metric',
        format_only=False,
        is_video_dataset=True)
    ]
test_evaluator = val_evaluator
'''

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
            'selective_modules': dict(lr_mult=1.0, decay_mult=1.0),
            'matching_module': dict(lr_mult=1.0, decay_mult=1.0),
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
        interval=val_interval))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
