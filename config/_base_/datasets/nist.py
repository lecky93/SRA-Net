# dataset settings
dataset_type = 'NistDataset'
# data_root = '/mnt/data1/qiangzeng/datasets/nist16'
data_root = r'D:\workspace\python\dataset\forgery\nist16'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='image',
        ann_dir='ann',
        split='train.txt',
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
        img_dir='image',
        ann_dir='ann',
        ann_suffix='_gt',
        # split='test.txt',
        is_train=False,
        img_size=(512, 512)))

test_dataloader = val_dataloader

val_evaluator = dict(type='AUCMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
