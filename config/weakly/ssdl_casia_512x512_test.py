custom_imports = dict(imports=['models', 'datasets', 'metrics'], allow_failed_imports=False)

_base_ = [
    '../_base_/datasets/nist.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

crop_size = (512, 512)
# data_preprocessor = dict(size=crop_size)
model = dict(
    type='SSDLNet'
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')


param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=8)

val_evaluator = dict(type='AUCMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

val_dataloader = dict(batch_size=8)
test_dataloader = val_dataloader