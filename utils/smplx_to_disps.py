import torch
import smplx
from pytorch3d.structures import Meshes


### Plot interactive scene for list of meshes and/or pointclouds
def smplx2disps(smplx_model, betas, scale, verts_disps=None):
    smplx_faces = torch.Tensor(smplx_model.faces.astype('int')).type(torch.int32).unsqueeze(0).cpu()

    init_verts = smplx_model.forward()['vertices'].cpu() * scale.cpu()
    init_mesh = Meshes(init_verts, smplx_faces)

    displaced_verts = smplx_model.forward(betas=betas)['vertices'].cpu() * scale.cpu()
    displaced_mesh = Meshes(displaced_verts, smplx_faces)

    if verts_disps is not None:
        displaced_verts += (displaced_mesh.verts_normals_packed() * verts_disps.cpu()).unsqueeze(0)
        displaced_mesh = Meshes(displaced_verts, smplx_faces)

    displacements = displaced_mesh.verts_packed() - init_mesh.verts_packed()
    displacements_along_nrm = torch.sum(displacements * init_mesh.verts_normals_packed(), dim=1).cpu()

    return displacements_along_nrm / scale.item()
