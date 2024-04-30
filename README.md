<div align="center">
  <a href=https://zju3dv.github.io/LoG_webpage/>
    <img src="assets/log_logo.svg" alt="logo" width="70%" align="center"/>
  </a>
</div>

---

![python](https://img.shields.io/github/languages/top/zju3dv/LoG)
![star](https://img.shields.io/github/stars/zju3dv/LoG)
[![license](https://img.shields.io/badge/license-zju3dv-white)](license)

**LoG** utilizes a single RTX 4090 for training highly realistic urban-scale models and for their real-time rendering. Visit our [**project page**](https://zju3dv.github.io/LoG_webpage/) for more demos.

https://github.com/chingswy/LoGvideos/assets/22812405/abb200c1-5b9c-48fd-a9a1-f235499153d2

Our code is built upon PyTorch and leverages [gaussian-splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization) techniques. 

## Quick Start

For a smooth setup, follow the [installation guide](./docs/install.md).

## Dataset Preparation

We employ [Colmap](https://colmap.github.io/) to prepare the dataset. Refer to the [preprocessing documentation](./docs/preprocess.md) for detailed instructions. A minimal example dataset is provided [here](https://forms.gle/E3Roi9zriu6Sk4557).

## Training

Training the model is as simple as one command:

```bash
python3 apps/train.py --cfg config/example/test/train_log.yml split train
```

We automatically configure heuristic parameters based on the dataset size.

We provide a path for interpolation visualization

```bash
python3 apps/train.py --cfg config/example/test/train_log.yml split demo_interpolate ckptname output/example/test/level_of_gaussian/model_init.pth
```

The visualization video will be stored at `output/example/test/level_of_gaussian/demo_interpolate/rgb.mp4`

## Immersive Visualization :rocket:

We will update a real-time rendering tool designed for immersive visualization.

## Acknowledgements

We acknowledge the following inspirational prior work:

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [LandMark](https://github.com/InternLandMark/LandMark)
- [UrbanBIS dataset](https://vcc.tech/UrbanBIS)
- [UrbanScene3D dataset](https://vcc.tech/UrbanScene3D)
- [UAVD4L dataset](https://github.com/RingoWRW/UAVD4L)

The rendering GUI is powered by our [EasyVolcap](https://github.com/zju3dv/EasyVolcap) tool.

Contributions are warmly welcomed! If you've made significant progress on any of these fronts, please consider submitting a pull request.


## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry.

```bibtex
@inproceedings{shuai2024LoG,
  title={Real-Time View Synthesis for Large Scenes with Millions of Square Meters},
  author={Shuai, Qing and Guo, Haoyu and Xu, Zhen and Lin, Haotong and Peng, Sida and Bao, Hujun and Zhou, Xiaowei},
  year={2024}
}
```