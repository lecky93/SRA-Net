# dataset settings
dataset_type = 'CopyMoveDataset'
data_root = '/data/sda/qiangzeng/datasets'
# data_root =  r'D:\workspace\python\dataset\forgery'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train+defacto.txt',
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
        ann_file='val.txt',
        is_train=False,
        img_size=(512, 512)))

test_dataloader = val_dataloader

val_evaluator = dict(type='MyIoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
