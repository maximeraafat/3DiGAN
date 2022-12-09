# 3DiGAN

> Code for **3D** aware **i**mplicit **G**enerative **A**dversial **N**etwork. Please mind the [remarks](#remarks) if you intend to make use of this code.

This repository extends a [lightweight generative network](https://github.com/lucidrains/lightweight-gan) to learn a distribution of 2D image UV textures wrapped on an underlying geometry, from a dataset of single-view photographs. Given a mesh prior, the generator synthesises UV appearance textures which are then rendered on top of the geometry. Colored points are sampled from the mesh and displaced along the mesh normal according to the last UV texture channel, which operates as a displacement map.

As stated above, this code builds on top of an implementation by GitHub user [lucidrains](https://github.com/lucidrains). The mentioned code license is provided in the below toggle.

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


## Installations

### Module dependencies

Clone this repository and install the dependencies with the below commands.
```bash
git clone https://github.com/maximeraafat/3DiGAN.git
pip install -r 3DiGAN/requirements.txt
```

The point-based rendering framework utilises [PyTorch3D](https://pytorch3d.org). Checkout the steps described in their provided [installation instruction set](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) with matching versions of **PyTorch**, and **CUDA** if applicable.

### SMPLX
Learning a human appearance model requires an underlying geometry prior : **3DiGAN** leverages the SMPLX parametric body model. Download the body model `SMPLX_NEUTRAL.npz` and corresponding UVs `smplx_uv.obj` from the [SMPLX project page](https://smpl-x.is.tue.mpg.de) into a shared folder. For training, we require a large collection of single view full body human images and their respective per image body parameters. Instead of storing the SMPLX underlying meshes individually, we store the body parameters for the full dataset into an **npz** file. Our code requires a specific structure for SMPLX, which we extract from the estimated body parameters with [PIXIE](https://github.com/YadiraF/PIXIE).

<details>
<summary> Details on how to get SMPLX parameters for <b>3DiGAN</b> with PIXIE  </summary>

Our code expects a **npz** file containing a list of 8 tensors : `['global_orient', 'body_pose', 'jaw_pose', 'left_hand_pose', 'right_hand_pose', 'expression', 'betas', 'cam']`. All per subject parameters are obtained from the PIXIE output in the following way.

```python
import numpy as np

params = np.load(<name>_param.pkl, allow_pickle=True)
prediction = np.load(<name>_prediction.pkl, allow_pickle=True)

global_orient = params['global_pose']
body_pose = params['body_pose']
jaw_pose = params['jaw_pose']
left_hand_pose = params['left_hand_pose']
right_hand_pose = params['right_hand_pose']
expression = params['exp'][:10]
betas = params['shape'][:10]
cam = prediction['cam']
```

`<name>_param.pkl` and `<name>_prediction.pkl` are the respective PIXIE outputs for a given image. Finally, the SMPLX parameters are concatenated together for all subjects in the training dataset of interest. For instance, the final global orientation shape will be `global_orient.shape = (num_subjects, 1, 3, 3)`, where the equivalent shape for one single SMPLX body is `(1, 3, 3)`. An example of SMPLX parameters extracted with PIXIE for version 1.0 of the [SHHQ](https://github.com/stylegan-human/StyleGAN-Human/blob/main/docs/Dataset.md) dataset, containing 40'000 images of high-quality full-body humans, is accessible [here](https://drive.google.com/file/d/1SoPnvbPv4oxuLJw3yP8J4-fnHuyRqTnj/view?usp=share_link).
</details>


## Training

Our code's purpose is the learning and synthesis of novel appearances; here we provide instructions for two different scenarios.

### Human Appearance

Given a large dataset of full body humans (see [SHHQ](https://github.com/stylegan-human/StyleGAN-Human/blob/main/docs/Dataset.md)) and corresponding SMPLX parameters, execute the following command.

```bash
python 3DiGAN/main.py --data <path/to/dataset> \
                      --models_dir <path/to/output/models> \
                      --results_dir <path/to/output/results> \
                      --name <run/name> \
                      --render \
                      --smplx_model_path <path/to/smplx>
```

The `--smplx_model_path` option provides the path to the SMPLX models folder, and requires an **npz** file containing all the estimated SMPLX parameters for each image in the dataset. See the [installations](#installations) section for details. The **npz** file must be accessible either by

1. renaming it to `dataset.npz` and including the file to the dataset folder under `<path/to/dataset>`, or by
2. providing the path to the **npz** file with `--labelpath <path/to/npz>`

### Arbitrary Geometry Appearance

To synthesise appearance for an arbitrary fixed geometry prior, provide the path to an **obj** mesh file containing UVs with `--mesh_obj_path`.

```bash
python 3DiGAN/main.py --data <path/to/dataset> \
                      --models_dir <path/to/output/models> \
                      --results_dir <path/to/output/results> \
                      --name <run/name> \
                      --render \
                      --mesh_obj_path <path/to/obj>
```

The `--mesh_obj_path` option requires a **json** file contaning estimated or ground truth camera azimuth and elevations for each image in the dataset. Note that the focal length to our point rendering camera is fixed to 10. Analogously to the human apperance modelling section, the **json** file must be accessible either by

1. renaming it to `dataset.json` and including the file to the dataset folder under `<path/to/dataset>`, or by
2. providing the path to the **json** file with `--labelpath <path/to/json>`

A toy dataset containing 2'000 renders of the PyTorch3D cow mesh with corresponding **json** file  comprising camera pose labels is accessible [here](https://drive.google.com/file/d/1xvLTY2hiVhkrYXl3UQxLDmpsotxvWExo/view?usp=share_link). The cow **obj** mesh file is accessible under [this link](https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj).


## Generation

To synthesise new human appearances from a trained generator, execute this command.

```bash
python 3DiGAN/main.py --generate \
                      --models_dir <path/to/output/models> \
                      --results_dir <path/to/output/results> \
                      --name <run/name> \
                      --render \
                      --labelpath <path/to/npz> \
                      --smplx_model_path <path/to/smplx>
```

Unlike for training, generation requires the `--labelpath` option since the dataset path is not provided. To synthesise arbitrary geometry appearances, replace the `--smplx_model_path` option for `--mesh_obj_path` and adapt `--labelpath`.

## Settings

This section discusses the relevant command line arguments. The code follows a similar structure to the original [lightweight GAN](https://github.com/lucidrains/lightweight-gan) implementation and supports the same options, while adding arguments for the rendering environment. Please visit the parent repository for further details.

* `--render_size` : square rendering resolution, by default set to `256`. This flag does not replace `--image_size` (also by default set to `256`), which is the generated square UV map resolution
* `--render` : whether to render the learned generated output. Without this flag, the code is essentially a copy of lightweight GAN
* `--renderer` : set by default to `default`, defines which point renderer to use. Has to be one of `default` or `pulsar`
* `--nodisplace` : call this flag to learn RGB appearances only, without a fourth displacement channel
* `--num_points` : number of points sampled from the underlying mesh geometry, by default set to `10**5`
* `--gamma` : point transparency coefficient for pulsar (defined between 1e-5 and 1), by default set to `1e-3`
* `--radius` : point radius, set by default to `0.01` for the default renderer and to `0.0005` for pulsar
* `--smplx_model_path` : path to the SMPLX models folder
* `--mesh_obj_path` : path to the underlying **obj** mesh file
* `--labelpath` : path to the **npz** file, respectively **json** file, containing the necessary SMPLX parameters or camera poses for rendering

Note that the generated UV textures are currently concatenated into 4 channel RGBD images rather than RGB images, plus a separate displacement texture map. The `--transparent` and `--greyscale` options are currently not supported when calling the `--render` flag.

Both the `--show_progress` and `--generate_interpolation` flags from the original parent implementation are functional, but operate in the UV image space rather than in the render space.


## Remarks

This code is my Master Thesis repository. Although fully operational, the generator does not converge due to many instabilities encountered during the fragile GAN training. Keep in mind that this implementation accordingly serves primarily as a basis for future research, rather than for direct usage. Further details to my work can be found on [my personal website](https://maximeraafat.github.io/projects/master_thesis).
