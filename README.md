# 3DiGAN

Master Thesis Repository, in progress...

TODO
* Write requirements.txt file

* Specifiy that this repo is built on top of Lightweight GAN.

* Provide instructions (and notebook) on how to obtain smplx parameters from pixie, and how they are stored in the .npz file

* Specifiy that if labelpath is provided, the data loader looks under path, othwerise it looks for file called dataset.json/dataset.npz within the training folder

* Specify requirements : mesh_obj_path requires a object file + dataset.json file with azimuth and elevations! focal length is fixed

* Specify sequirements : smplx_model_path requires smplx files + dataset.npz file with parameters extracted as explained above

* Test generate and generate_interpolation with and without displacements + show_progress and calculate_fid functions

* TODO : split fourth displacement channel from RGB channels when saving textures

Calculate FID is done in rendering space if flag is called
Show progress and generate interpolation do no support rendering
For now, UV RGBD textures are stored as 4 channel png images

## Training Setting
### Training Options

## Generation