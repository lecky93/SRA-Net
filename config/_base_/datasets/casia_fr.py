# dataset settings
dataset_type = 'CasiaDataset'
data_root = '/mnt/data1/qiangzeng/datasets/casia'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='CASIA2',
        ann_dir='CASIA2',
        ann_file='train_fr.txt',
        ann_suffix='_gt',
        is_train=True,
        img_size=(512, 512)))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='COVERAGE/image',
        # ann_dir='COVERAGE/mask',
        img_dir='CASIA1/image',
        ann_dir='CASIA1/ann',
        ann_suffix='_gt',
        is_train=False,
        img_size=(512, 512)))

test_dataloader = val_dataloader

val_evaluator = dict(type='MyIoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
