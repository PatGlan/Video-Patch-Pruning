_base_ = ['./youtube_vis_2019_vpp.py']


dataset_type = 'YouTubeVISDataset'
data_root = 'data/youtube_vis_2021/'
dataset_version = data_root[-5:-1]  # 2019 or 2021
# dataloader
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2021_train.json',
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2021_valid.json',
    )
)
test_dataloader = val_dataloader
