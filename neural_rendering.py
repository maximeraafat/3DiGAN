try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

import torch
import smplx
import numpy as np
from typing import List
from torchvision.io import read_image

from utils.smpl_to_smplx import smpl2smplx
from utils.camera_calibration import get_camera_parameters
from utils.get_renderers import get_renderers
from utils.pointrend_segmentation import get_pointrend_segmentation

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesUV
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Initialize smplx parameters + displacements given an smplx.SMPLXLayer object
def get_init_mesh(smplx_model, requires_grad=True, device:torch.device=device):
    global_orient = torch.nn.Parameter( torch.Tensor([[0, 0, 0]]).to(device), requires_grad=requires_grad )
    transl = torch.nn.Parameter( torch.Tensor([[0, 0, 0]]).to(device), requires_grad=requires_grad )
    body_pose = torch.nn.Parameter( torch.zeros([1, 21, 3]).to(device), requires_grad=requires_grad )
    left_hand_pose = torch.nn.Parameter( torch.zeros([1, 15, 3]).to(device), requires_grad=requires_grad )
    right_hand_pose = torch.nn.Parameter( torch.zeros([1, 15, 3]).to(device), requires_grad=requires_grad )
    jaw_pose = torch.nn.Parameter( torch.zeros([1, 1, 3]).to(device), requires_grad=requires_grad )
    expression = torch.nn.Parameter( torch.zeros([1, 10]).to(device), requires_grad=requires_grad )
    betas = torch.nn.Parameter( torch.zeros([1, 10]).to(device), requires_grad=requires_grad )
    scale = torch.nn.Parameter( torch.Tensor([1.0]).to(device), requires_grad=requires_grad )

    num_smplx_verts = smplx_model.get_num_verts()
    verts_disps = torch.nn.Parameter( torch.zeros([num_smplx_verts, 1]).to(device), requires_grad=requires_grad )

    return global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, verts_disps


### Given smplx parameters + displacements (optional), construct corresponding mesh
def construct_textured_mesh(smplx_model, texture_uv, global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, verts_disps=None, device:torch.device=device):
    smplx_faces = torch.Tensor(smplx_model.faces.astype('int')).type(torch.int32).unsqueeze(0).to(device)

    smplx_verts = smplx_model.forward(global_orient=axis_angle_to_matrix(global_orient),
                                      body_pose=axis_angle_to_matrix(body_pose),
                                      left_hand_pose=axis_angle_to_matrix(left_hand_pose),
                                      right_hand_pose=axis_angle_to_matrix(right_hand_pose),
                                      jaw_pose=axis_angle_to_matrix(jaw_pose),
                                      expression=expression, betas=betas)['vertices'].to(device)

    smplx_mesh = Meshes(smplx_verts * scale + transl, smplx_faces, texture_uv)

    if verts_disps is not None:
        verts_smplx_disp = (smplx_verts * scale) + (smplx_mesh.verts_normals_packed() * verts_disps).unsqueeze(0)
        smplx_mesh = Meshes(verts_smplx_disp + transl, smplx_faces, texture_uv)

    return smplx_mesh


### Neural rendering
def neural_renderer(smplx_model, subject:int, poses:List[str], iterations:int, smplx_uv_path:str, textures_path:str, rescale_factor:int=2):
    obj_mesh = load_obj(smplx_uv_path, load_textures=False)
    faces_uvs = obj_mesh[1].textures_idx.unsqueeze(0).to(device)
    verts_uvs = obj_mesh[2].verts_uvs.unsqueeze(0).to(device)

    texture_path = textures_path + 'median_subject_%d.png' % subject
    texture = read_image(texture_path)
    texture = torch.moveaxis(texture, 0, 2).unsqueeze(0).to(device).float() * 1.0/255

    img_size = (1080, 1920) # photo resolution
    render_res = ( int(1080/rescale_factor), int(1920/rescale_factor) ) # render resolution

    betas_list = []
    scale_list = []
    verts_disps_list = []
    texture_list = []
    total_losses = []

    for pose in poses:
        print('fit parameters for pose %d out of %d\n' % (i+1, len(poses)) )
        print('fit new smplx model to provided humbi smpl parameters\n')
        global_orient, transl, body_pose, betas, scale, pose_loss, shape_loss = smpl2smplx(smplx_model, subject, pose, pose_iterations=200, shape_iterations=100)
        left_hand_pose, right_hand_pose, jaw_pose, expression = get_init_mesh(smplx_model)[3:7]
        verts_disps = get_init_mesh(smplx_model)[-1]

        opt_smplx = torch.optim.Adam([global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale], lr=1.0e-3)
        opt_geom = torch.optim.Adam([body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, verts_disps], lr=1.0e-3)
        opt_txt = torch.optim.Adam([texture], lr=1.0e-2)

        sched_smplx = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_smplx, patience=10, verbose=True)
        sched_geom = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_geom, patience=10, verbose=True)
        sched_txt = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_txt, patience=10, verbose=True)

        l1_loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()

        pbar = tqdm(total = iterations * 107)
        for i in range(iterations):
            total_loss = 0

            for camera_idx in np.random.choice(107, 107, replace=False):
                # Construct mesh
                texture_uv = TexturesUV(maps=texture, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
                mesh = construct_textured_mesh(smplx_model, texture_uv, global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, verts_disps)

                # Extract camera parameters and construct camera
                R, T, f, p = get_camera_parameters(subject, camera_idx)
                cameras = PerspectiveCameras(focal_length=-f, principal_point=p, R=R, T=T, in_ndc=False, image_size=(img_size,), device=device)

                # Render mesh from camera viewpoint
                silhouette_renderer, phong_renderer = get_renderers(cameras, render_res, device=device)
                phong_render = phong_renderer(mesh).detach()
                silhouette_render = silhouette_renderer(mesh).detach()

                rgb_render = phong_render[0, ..., :3]
                silh_render = silhouette_render[0, ..., 3]

                # Segment person in photo from camera viewpoint
                photo_path = 'subject_%s/body/%s/image/image%s.jpg' % (subject, pose, str(camera_idx).zfill(7))
                photo, silh_photo, rgb_photo = get_pointrend_segmentation(photo_path, device=device)

                photo = photo[::rescale_factor, ::rescale_factor].to(device)
                silh_photo = silh_photo[0, ::rescale_factor, ::rescale_factor].float().to(device)
                rgb_photo = rgb_photo[0, ::rescale_factor, ::rescale_factor].to(device)

                # Penalize difference between render and image segmentation
                loss = l1_loss(rgb_photo, rgb_render) + l1_loss(silh_photo, silh_render) + mse_loss(rgb_photo, rgb_render) + mse_loss(silh_photo, silh_render) + 0.01*torch.norm(verts_disps)
                total_loss += float(loss)

                opt_geom.zero_grad()
                opt_txt.zero_grad()
                pbar.set_description('loss = %.6f' % loss)
                loss.backward()
                opt_txt.step()
                opt_geom.step()
                pbar.update(1)

            print('total loss = %.6f for pose %s\n' % (total_loss, pose))
            sched_geom.step(total_loss)
            sched_txt.step(total_loss)

        betas_list.append(betas)
        scale_list.append(scale)
        verts_disps_list.append(verts_disps)
        texture_list.append(texture)
        total_losses.append(total_loss)

    return betas_list, scale_list, verts_disps_list, texture_list, total_losses
