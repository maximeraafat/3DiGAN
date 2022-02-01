# GTA : Generated Textured Avatars

**GTA** is a method for synthetizing virtual humans from a dataset of UV textures, which currently consists of two main features.

1. [SMPL-X](https://github.com/vchoutas/smplx) UV texture reconstruction for subjects in [HUMBI](https://github.com/zhixuany/HUMBI) dataset
2. Generating new color textures with [Lightweight GAN](https://github.com/lucidrains/lightweight-gan) given the reconstructed UV textures


## Demos

Details on how to use **GTA** will soon be provided here.


## Installation

Run the following command to install most requirements
```bash
$ pip install git+https://github.com/maximeraafat/gta
```
and install the below packages
- PyTorch3D : https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- Detectron2 : https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md

Additionally, clone [Detectron2](https://github.com/facebookresearch/detectron2) (if not already done) inside of this repository (```gta/```)
```bash
$ git clone https://github.com/facebookresearch/detectron2.git detectron2_repo
```

## License
This library is licensed under the MIT License. See the [LICENSE](LICENSE) file.

[//]: # (TODO: ## Structure)
