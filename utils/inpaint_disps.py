import torch
import numpy as np

from scipy import interpolate
from torchvision.io import read_image

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Extract vertex uv pixel positions on a 2D square map
def verts_uvs_positions(smplx_uv_path:str, map_size:int=1024):
    # See https://github.com/facebookresearch/pytorch3d/discussions/588
    smplx_uv_mesh = load_obj(smplx_uv_path, load_textures=False)

    nb_verts = smplx_uv_mesh[0].shape[0]

    flatten_verts_idx = smplx_uv_mesh[1].verts_idx.flatten().to(device)
    flatten_textures_idx = smplx_uv_mesh[1].textures_idx.flatten().to(device)
    verts_uvs = smplx_uv_mesh[2].verts_uvs.to(device)

    verts_to_uv_index = torch.zeros(nb_verts, dtype=torch.int64).to(device)
    verts_to_uv_index[flatten_verts_idx] = flatten_textures_idx
    verts_to_uvs = verts_uvs[verts_to_uv_index]

    uv_x = ( float(map_size) * verts_to_uvs[:,0] ).unsqueeze(0).to(device)
    uv_y = ( float(map_size) * (1.0 - verts_to_uvs[:,1]) ).unsqueeze(0).to(device)
    verts_uvs_positions = torch.cat((uv_x, uv_y)).moveaxis(0,1).round().to(device)

    return verts_uvs_positions


### Create displacement map for each vertex and perform interpolation (inpaining) between vertex values
def inpaint_disps(subject:int, displacements:torch.Tensor, smplx_uv_path:str, path_to_textures:str, mask_disps:bool=False):
    texture = read_image(path_to_textures + 'median_subject_%d.png' % subject)
    texture = torch.moveaxis(texture, 0, 2).to(device)
    map_size = texture.shape[:2]

    verts_uvs = verts_uvs_positions(smplx_uv_path, map_size[0]).flip(1)

    mask = (texture[:,:,0] == 0) & (texture[:,:,1] == 0) & (texture[:,:,2] == 0)

    interp = interpolate.LinearNDInterpolator(points=verts_uvs.cpu(), values=displacements.detach().cpu().numpy(), fill_value=0)
    inpainted_displacements = interp( list(np.ndindex(map_size)) ).reshape(map_size)

    if mask_disps:
        inpainted_displacements[mask.cpu()] = 0

    return torch.Tensor(inpainted_displacements).cpu(), ~mask.cpu(), texture.cpu()
