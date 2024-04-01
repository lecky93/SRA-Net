# dataset settings
dataset_type = 'NistDataset'
# data_root = '/mnt/data1/qiangzeng/datasets'
data_root = r'D:\workspace\python\dataset\forgery'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='casia/CASIA2/image',
        ann_dir='casia/CASIA2/ann',
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
        # img_dir='test',
        # ann_dir='nist16/ann',
        img_dir='casia/CASIA1/image',
        ann_dir='casia/CASIA1/ann',
        ann_suffix='_gt',
        is_train=False,
        img_size=(512, 512)))

test_dataloader = val_dataloader

val_evaluator = dict(type='AUCMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
