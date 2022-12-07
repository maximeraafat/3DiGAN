# 3DiGAN

This repository extends a [lightweight](https://github.com/lucidrains/lightweight-gan) generative adversial network (GAN) to generate 2D image UV textures on top of an underlying geometry. Given a mesh prior, the generator synthesises UV appearance textures which are then rendered on top of the geometry. Colored points are sampled from the mesh and displaced along the mesh normal according to the last UV texture channel, which operates as a displacement map.

## Installations

Clone this repository and install the dependencies with the below commands.
```bash
git clone https://github.com/maximeraafat/3DiGAN.git
pip install -r 3DiGAN/requirements.txt
```

The point-based rendering framework utilises [PyTorch3D](https://pytorch3d.org). Checkout the steps described in their provided [installation instruction set](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for matching versions of **PyTorch** and **CUDA**.


## Training

### SMPL-X

### Arbitrary Mesh


For now, UV RGBD textures are stored as 4 channel png images

If labelpath is provided, the data loader looks under path, othwerise it looks for file called dataset.json/dataset.npz within the training folder

mesh_obj_path requires a object file + dataset.json file with azimuth and elevations! focal length is fixed

smplx_model_path requires smplx files + dataset.npz file with parameters extracted as explained above

## Generation

## Settings

Calculate FID is done in rendering space if flag is called
Show progress and generate interpolation do no support rendering


As stated above, 3DiGAN is built on top of an implementation by GitHub user [lucidrains](https://github.com/lucidrains). The mentioned code license is provided in the below toggle.

<details>
<summary> <b>Lightweight GAN</b> license </summary>

```markdown
MIT License

Copyright (c) 2021 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
</details>

TODO
* Write requirements.txt file
* Provide instructions (and notebook) on how to obtain smplx parameters from pixie, and how they are stored in the .npz file
