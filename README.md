# Anatomy-Informed Cephalometric X-ray Generation(AICG)

This is the official repository of our paper ***Towards Better Cephalometric Landmark Detection with DiffusionData Generation***.

More details are available in [Project website](https://um-lab.github.io/cepha-generation/)

## News
[2025.03.05] Our paper has been accepted at IEEE Transactions on Medical Imaging! 🎉

## Quick start
### 1. Conda Env Preparation
To build a compatible conda env, you only need to run the following lines one by one:

```bash
conda create -n aicg python=3.10
conda activate aicg
pip install -r requirements.txt
pip install -U openmim
cd mmpose_package/mmpose
pip install -e .
mim install mmengine
mim install "mmcv>=2.0.0"
pip install --upgrade numpy
```

### 2. Test
Test with the pretrained weights:
```bash
python test_and_visualize.py --config 'configs/your_config_file' --checkpoint 'path_to_your_checkpoint'
```

## Citation
```latex
@article{guo2025towards,
  title={Towards Better Cephalometric Landmark Detection with Diffusion Data Generation},
  author={Guo, Dongqian and Han, Wencheng and Lyu, Pang and Zhou, Yuxi and Shen, Jianbing},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```