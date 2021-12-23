try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

import os
import torch
import smplx
import numpy as np
from torchvision.io import read_image

from utils.smpl_to_smplx import smpl2smplx
from utils.camera_calibration import get_camera_parameters
from utils.renderers import get_renderers
from utils.pointrend_segmentation import get_pointrend_segmentation

from pytorch3d.io import load_obj, save_obj
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

    texture = torch.nn.Parameter( torch.zeros([1, 2048, 2048, 3]).to(device), requires_grad=requires_grad )

    return global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, verts_disps, texture


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
def neural_renderer(smplx_model, subject:int, pose:str, iterations:int, smplx_uv_path:str, rescale_factor:int=3, save_path:str=None):
    ## Segment all photos
    camera_idx_list = []
    silh_photo_list = []
    rgb_photo_list = []
    print('segment all photos before neural rendering')
    cam_loop = tqdm(np.random.choice(107, 107, replace=False), total = 107)
    for camera_idx in cam_loop:
        try:
            # Check if camera exists
            get_camera_parameters(subject, camera_idx)
        except:
            print('camera with index %d does not exist' % camera_idx)
            continue

        # Segment person in photo from camera viewpoint
        photo_path = 'subject_%s/body/%s/image/image%s.jpg' % (subject, pose, str(camera_idx).zfill(7))
        photo, silh_photo, rgb_photo = get_pointrend_segmentation(photo_path, device=device)

        silh_photo = silh_photo[0, ::rescale_factor, ::rescale_factor].float().to(device)
        rgb_photo = rgb_photo[0, ::rescale_factor, ::rescale_factor].to(device)

        camera_idx_list.append(camera_idx)
        silh_photo_list.append(silh_photo)
        rgb_photo_list.append(rgb_photo)

    # uv coordinates
    obj_mesh = load_obj(smplx_uv_path, load_textures=False)
    faces_uvs = obj_mesh[1].textures_idx.unsqueeze(0).to(device)
    verts_uvs = obj_mesh[2].verts_uvs.unsqueeze(0).to(device)

    img_size = (1080, 1920) # photo resolution
    render_res = ( int(1080/rescale_factor), int(1920/rescale_factor) ) # render resolution

    ## SMPL fitting + Neural rendering
    print('fit new smplx model to provided humbi smpl parameters')
    global_orient, transl, body_pose, betas, scale, pose_loss, shape_loss = smpl2smplx(smplx_model, subject, pose, pose_iterations=200, shape_iterations=100)
    left_hand_pose, right_hand_pose, jaw_pose, expression = get_init_mesh(smplx_model)[3:7]
    verts_disps, texture = get_init_mesh(smplx_model)[-2:]

    opt_pose_shape = torch.optim.Adam([body_pose, betas], lr=0.01)
    opt_geom = torch.optim.Adam([verts_disps], lr=0.01)
    opt_txt = torch.optim.Adam([texture], lr=0.01)

    sched_pose_shape = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_pose_shape, patience=10, threshold=1, verbose=True)
    sched_geom = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_geom, patience=10, threshold=1, verbose=True)
    sched_txt = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_txt, patience=10, threshold=1, verbose=True)

    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()

    print('neural rendering for pose')
    loop = tqdm(total = iterations * len(camera_idx_list))
    for i in range(iterations):
        total_loss = 0
        for k, camera_idx in enumerate(camera_idx_list):
            # Extract camera parameters
            R, T, f, p = get_camera_parameters(subject, camera_idx)

            # Construct camera
            cameras = PerspectiveCameras(focal_length=-f, principal_point=p, R=R, T=T, in_ndc=False, image_size=(img_size,), device=device)

            # Construct mesh
            texture_output = torch.clamp(texture, min=0, max=1)
            verts_disps_output = torch.clamp(verts_disps, min=0)
            texture_uv = TexturesUV(maps=texture_output, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
            mesh = construct_textured_mesh(smplx_model, texture_uv, global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, verts_disps_output)

            # Render mesh from camera viewpoint
            silhouette_renderer, phong_renderer = get_renderers(cameras, render_res, device=device)
            phong_render = phong_renderer(mesh)
            silhouette_render = silhouette_renderer(mesh)

            rgb_render = phong_render[0, ..., :3]
            silh_render = silhouette_render[0, ..., 3]

            # Compute loss
            loss = 10* ( l1_loss(rgb_photo_list[k], rgb_render) + l1_loss(silh_photo_list[k], silh_render) )
            loss += 5 * ( mse_loss(rgb_photo_list[k], rgb_render) + mse_loss(silh_photo_list[k], silh_render) )
            loss += 0.1 * torch.linalg.norm(verts_disps)
            total_loss += float(loss)

            # Backpropagate loss
            opt_pose_shape.zero_grad()
            opt_geom.zero_grad()
            opt_txt.zero_grad()
            loop.set_description('neural rendering loss = %.6f' % loss)
            loss.backward()
            opt_pose_shape.step()
            opt_geom.step()
            opt_txt.step()
            loop.update(1)

        print('neural rendering total loss for iteration %d : %.6f' % ((i+1), total_loss))
        sched_pose_shape.step(total_loss)
        sched_geom.step(total_loss)
        sched_txt.step(total_loss)

        if save_path is not None and i%2 == 0 and i > 0:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, 'mesh_subj_%d_pose_%s_iter_%d.obj' % (subject, pose, i))
            save_obj(filename, verts=mesh.verts_packed(), faces=mesh.faces_packed(), verts_uvs=verts_uvs[0], faces_uvs=faces_uvs[0], texture_map=texture_output[0])

    geometry = global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, verts_disps_output

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, 'final_mesh_subj_%d_pose_%s.obj' % (subject, pose))
        save_obj(filename, verts=mesh.verts_packed(), faces=mesh.faces_packed(), verts_uvs=verts_uvs[0], faces_uvs=faces_uvs[0], texture_map=texture_output[0])

    return geometry, texture_output.detach()
