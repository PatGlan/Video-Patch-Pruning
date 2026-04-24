_base_ = [
    './vpp_vitSmall_4xb2_6e_0.55PKR_ytvis21.py'
]

work_dir = 'output/vpp_vitSmall_4xb2_6e_0.4PKR'

#model
model = dict(
    detector=dict(
        type='Mask2Former',
        backbone=dict(
            type='SelectiveViTAdapter',
            selec_skip_ids=[0],
            pruning_method = 'ivpp',

            #selective moduels
            selective_module=dict(
                type='SelectiveModule',
                emb_dim=192,
                hidden_dim=192//4,
                tau=10, #controlls gumbel noise
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
        ),
    ),
)


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
