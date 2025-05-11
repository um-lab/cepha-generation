default_scope = 'mmpose'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=50,
        save_best='SDR 2.0mm',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))
custom_hooks = [dict(type='SyncBuffersHook')]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')

log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False

backend_args = dict(backend='local')

train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# optimizer
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]


# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)


# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(512, 512), heatmap_size=(128, 128), sigma=2)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(512, 512),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.55,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/data2/jianbingshen/guodongqian/CL-Detection2023-MMPose/mae_pretrain_vit_base_20230913.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=38,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))


dataset_type = 'CephalometricDataset'
data_mode = 'topdown'
data_root = '/data2/jianbingshen/guodongqian/diffusers/examples/controlnet/'
# data_root = '/mnt/guodongqian/CL-Detection2023-MMPose'

# meta_keys is used to add 'spacing' information, please do not change it if you don't know its usage
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'category_id', 'crowd_index', 'ori_shape',
                                           'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
                                           'flip_direction', 'flip_indices', 'raw_ann_info', 'spacing'))
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'category_id', 'crowd_index', 'ori_shape',
                                           'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
                                           'flip_direction', 'flip_indices', 'raw_ann_info', 'spacing'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='/data2/jianbingshen/guodongqian/CL-Detection2023-MMPose/generate_condition/annotations/train_gen20percent.json',
        data_prefix=dict(img='generated_images/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=25,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='/data2/jianbingshen/guodongqian/CL-Detection2023-MMPose/generate_condition/annotations/all_corrected.json',
        data_prefix=dict(img='MMPose/'),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=25,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='/data2/jianbingshen/guodongqian/CL-Detection2023-MMPose/generate_condition/annotations/test.json',
        # data_prefix=dict(img='cepha_data/images/'),
        data_prefix=dict(img='MMPose/'),
        test_mode=True,
        pipeline=val_pipeline))

val_evaluator = dict(
    type='CephalometricMetric')
test_evaluator = val_evaluator


