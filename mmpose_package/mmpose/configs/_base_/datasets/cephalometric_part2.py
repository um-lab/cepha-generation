# !/usr/bin/env python
# -*- coding:utf-8 -*-

dataset_info = dict(
    dataset_name='cephalometric',
    challenge_info=dict(
        author='Zhang, Hongyuan and Huang, Bingsheng',
        title='Cephalometric Landmark Detection in Lateral X-ray Images 2023',
        container='Medical Image Computing and Computer Assisted Intervention Conference',
        year='2023',
        homepage='https://cl-detection2023.grand-challenge.org/',
    ),
    keypoint_info={
        0: {'name': '0', 'id': 0, 'color': [255, 0, 0], 'type': '', 'swap': ''},
        1: {'name': '1', 'id': 1, 'color': [255, 0, 0], 'type': '', 'swap': ''},
        2: {'name': '2', 'id': 2, 'color': [255, 0, 0], 'type': '', 'swap': ''},
        3: {'name': '3', 'id': 3, 'color': [255, 0, 0], 'type': '', 'swap': ''},
        4: {'name': '4', 'id': 4, 'color': [255, 0, 0], 'type': '', 'swap': ''},
        5: {'name': '5', 'id': 5, 'color': [255, 0, 0], 'type': '', 'swap': ''},
    },
    skeleton_info={},
    joint_weights=1,
    sigmas=[1] * 38
)
