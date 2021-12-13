try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

import torch
import smplx
import numpy as np
from bps import bps
from typing import List

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.loss import chamfer_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    from chamferdist import ChamferDistance

### Construct pointcloud
def construct_pointcloud(subject:int, pose:str, normalize:bool=False, device:torch.device=device):
    filename = 'subject_%d/body/%s/reconstruction/surface_reconstruction.txt' % (subject, pose)
    reconstruction = np.loadtxt(filename)

    if normalize:
        reconstruction = bps.normalize(np.expand_dims(reconstruction[:, 0:3], 0))
        reconstruction = torch.Tensor(reconstruction).to(device)
    else:
        reconstruction = torch.Tensor(reconstruction[:, 0:3]).unsqueeze(0).to(device)

    # rotate_180_z = torch.Tensor([[-1, 0,  0], [ 0, -1, 0], [ 0,  0, 1]]).to(device)
    # rotate_180_y = torch.Tensor([[-1, 0,  0], [ 0, 1,  0], [ 0, 0, -1]]).to(device)
    points = reconstruction # @ rotate_180_z @ rotate_180_y
    pointcloud = Pointclouds(points)

    return pointcloud


### Construct pointcloud list
def pointcloud_list(subject:int, poses:List[str], device:torch.device=device):
    pointclouds = []

    for pose in tqdm(poses):
        poincloud = construct_pointcloud(subject, pose, False, device)
        pointclouds.append(poincloud)

    return pointclouds


### Extract smpl parameters, and store into smplx compatible parameters tensors
def extract_smpl_param(subject:int, pose:str, device:torch.device=device):
    filename = 'subject_%d/body/%s/reconstruction/smpl_parameter.txt' % (subject, pose)
    smpl_param = np.loadtxt(filename)

    ## See https://github.com/zhixuany/HUMBI#body--cloth for how to extract parameters
    ## and https://issueexplorer.com/issue/facebookresearch/frankmocap/91 to see smpl pose to smplx

    scale = torch.Tensor([smpl_param[0]]).to(device) # smpl and smplx compatible
    transl = torch.Tensor(smpl_param[1:4]).to(device) # smpl and smplx compatible (no perfect match)
    global_orient = torch.Tensor([smpl_param[4:7]]).to(device) # smpl and smplx compatible (no perfect match)
    # body_pose = torch.Tensor(smpl_param[7:76]).reshape(1, 23, 3).to(device) # only smpl compatible
    body_pose_smplx = torch.Tensor(smpl_param[7:70]).reshape(1, 21, 3).to(device) # smplx compatible (no perfect match)
    # betas = torch.Tensor([smpl_param[76:]]).to(device) # only smpl compatible

    return scale, transl, global_orient, body_pose_smplx


### Initialize smplx parameters + displacements given an smplx.SMPLXLayer object
def get_init_smplx(smplx_model, requires_grad=True, device:torch.device=device):
    num_smplx_verts = smplx_model.get_num_verts()

    global_orient = torch.nn.Parameter( torch.Tensor([[0, 0, 0]]).to(device), requires_grad=requires_grad )
    transl = torch.nn.Parameter( torch.Tensor([[0, 0, 0]]).to(device), requires_grad=requires_grad )
    body_pose = torch.nn.Parameter( torch.zeros([1, 21, 3]).to(device), requires_grad=requires_grad )
    right_hand_pos = torch.nn.Parameter( torch.zeros([1, 15, 3]).to(device), requires_grad=requires_grad )
    left_hand_pos = torch.nn.Parameter( torch.zeros([1, 15, 3]).to(device), requires_grad=requires_grad )
    expression = torch.nn.Parameter( torch.zeros([1, 10]).to(device), requires_grad=requires_grad )
    betas = torch.nn.Parameter( torch.zeros([1, 10]).to(device), requires_grad=requires_grad )

    scale = torch.nn.Parameter( torch.Tensor([1.0]).to(device), requires_grad=requires_grad )
    verts_disps = torch.nn.Parameter( torch.zeros([num_smplx_verts, 1]).to(device), requires_grad=requires_grad )

    return global_orient, transl, body_pose, betas, scale, verts_disps


### Sample points from smplx model with provided parameters
def get_smplx_points(smplx_model, global_orient, transl, body_pose, betas, scale, verts_disps=None):

    smplx_faces = torch.Tensor(smplx_model.faces.astype('int')).type(torch.int32).unsqueeze(0).to(device)

    verts_smplx = smplx_model.forward(global_orient=axis_angle_to_matrix(global_orient),
                                      body_pose=axis_angle_to_matrix(body_pose),
                                      betas=betas)['vertices'].to(device)

    verts_smplx *= scale
    verts_smplx += transl
    mesh_smplx = Meshes(verts_smplx, smplx_faces)

    # if verts_disps is not None, we additionally displace the vertices along the normal
    if verts_disps is not None:
        verts_smplx_disp = verts_smplx + (mesh_smplx.verts_normals_packed() * verts_disps.clamp(min=0.0)).unsqueeze(0)
        mesh_smplx = Meshes(verts_smplx_disp, smplx_faces)

    sampled_smplx_points = sample_points_from_meshes(mesh_smplx, num_samples=10**3)

    return sampled_smplx_points, mesh_smplx


### Optimization loop for smplx parameters + displacements
def optimization_loop(smplx_model, pcl:Pointclouds, iters:int, opt, sched, global_orient, transl, body_pose, betas, scale, verts_disps):

    if verts_disps is not None:
        zeros = torch.zeros(verts_disps.shape).to(device)

    if torch.cuda.is_available():
        cdist = ChamferDistance()

    loop = tqdm(range(iters), total = iters)
    for i in loop:
        sample_deformed_mesh, deformed_mesh = get_smplx_points(smplx_model, global_orient, transl, body_pose, betas, scale, verts_disps)

        if torch.cuda.is_available():
            loss = cdist(sample_deformed_mesh, pcl.points_packed().unsqueeze(0), bidirectional=True)
        else:
            loss_forward, _ = chamfer_distance(pcl, sample_deformed_mesh)
            loss_backward, _ = chamfer_distance(sample_deformed_mesh, pcl)
            loss = (loss_forward + loss_backward) * 10000

            '''
            if verts_disps is not None:
                # penalize if verts_disps has negative displacements
                loss += torch.sum( torch.where(verts_disps >= 0, zeros, torch.abs(verts_disps)) ).item()
            '''

        opt.zero_grad()
        loop.set_description('total_loss = %.6f' % loss)
        loss.backward()
        opt.step()
        sched.step(loss)

    return global_orient, transl, body_pose, betas, scale, verts_disps, loss, deformed_mesh


### Fit smplx parameters + displacements to pointcloud
def fit_pcl_pose(smplx_model, subject:int, pose:str, pcl:Pointclouds, global_iters:int, shape_iters:int):

    # we initiliaze the smplx parameters to learn, but ignore body_pose and scale since we get them from the humbi smpl paramaters
    global_orient, transl, body_pose, betas, scale, verts_disps = get_init_smplx(smplx_model)
    scale, transl, global_orient, body_pose = extract_smpl_param(subject, pose)
    scale = torch.nn.Parameter(scale, requires_grad=True)
    transl = torch.nn.Parameter(transl, requires_grad=True)
    global_orient = torch.nn.Parameter(global_orient, requires_grad=True)
    body_pose = torch.nn.Parameter(body_pose, requires_grad=True)

    # global optimizer : optimizes smplx parameters
    global_optimizer = torch.optim.Adam([global_orient, transl, body_pose, betas, scale], lr=0.01)
    global_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(global_optimizer, patience=20, verbose=True)

    # shape optimizer : optimizes shape and displacements
    shape_optimizer = torch.optim.Adam([global_orient, transl, body_pose, betas], lr=0.001)
    shape_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(shape_optimizer, patience=20, verbose=True)

    global_output = optimization_loop(smplx_model, pcl, global_iters, global_optimizer, global_scheduler, global_orient, transl, body_pose, betas, scale, None)
    global_orient, transl, body_pose, betas, scale, _, global_loss, global_deformed_mesh = global_output

    shape_output = optimization_loop(smplx_model, pcl, shape_iters, shape_optimizer, shape_scheduler, global_orient, transl, body_pose, betas, scale, verts_disps)
    global_orient, transl, body_pose, betas, scale, verts_disps, shape_loss, shape_deformed_mesh = shape_output

    displacements_along_nrm = displacement_from_smplx_param(smplx_model, betas, scale, verts_disps)

    return displacements_along_nrm, global_deformed_mesh, shape_deformed_mesh, global_loss, shape_loss


### Fit smplx shape parameters + displacements to all specified poses
def fit_pointclouds(smplx_model, subject:int, poses:List[str], pointclouds:List[Pointclouds], global_iterations:int, shape_iterations:int):
    displacements = []
    global_visualizations = []
    shape_visualizations = []
    global_losses = [] # should be similar in every index
    shape_losses = [] # should be similar in every index

    if isinstance(poses, str): # if poses is just a single pose
        poses = [poses]

    for i, (pcl, pose) in enumerate(zip(pointclouds, poses)):
        print('fit parameters for pointcloud %d out of %d' % (i+1, len(pointclouds)) )
        output = fit_pcl_pose(smplx_model, subject, pose, pcl, global_iterations, shape_iterations)

        displacements_along_nrm, global_deformed_mesh, shape_deformed_mesh, global_loss, shape_loss = output

        displacements.append(displacements_along_nrm)
        global_visualizations.append([pcl, global_deformed_mesh])
        shape_visualizations.append([pcl, shape_deformed_mesh])
        global_losses.append(global_loss)
        shape_losses.append(shape_loss)

    return displacements, global_visualizations, shape_visualizations, global_losses, shape_losses


### Extract the displacement along the normal for the specified shape parameters
def displacement_from_smplx_param(smplx_model, betas, scale, verts_disps):
    smplx_faces = torch.Tensor(smplx_model.faces.astype('int')).type(torch.int32).unsqueeze(0).to(device)

    init_verts = smplx_model.forward()['vertices'].to(device) * scale
    init_mesh = Meshes(init_verts, smplx_faces)

    displaced_verts = smplx_model.forward(betas=betas)['vertices'].to(device) * scale
    displaced_verts = displaced_verts + (init_mesh.verts_normals_packed() * verts_disps).unsqueeze(0)
    mesh_smplx_disp = Meshes(displaced_verts, smplx_faces)

    displacements = mesh_smplx_disp.verts_packed() - init_mesh.verts_packed()
    displacements_along_nrm = torch.sum(displacements * init_mesh.verts_normals_packed(), dim=1).to(device)

    return displacements_along_nrm
