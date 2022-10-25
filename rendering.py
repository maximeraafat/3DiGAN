import os
import smplx
import torch
import numpy as np

from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import TexturesUV, TexturesVertex

from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PulsarPointsRenderer
)

# TODO
# normalize vertices by centering and scaling
def normalize_verts(verts):
    mean = verts.mean(axis=1) # vertices mean
    max = (verts - mean).square().sum(axis=2).sqrt().max() # largest variation from mean
    return (verts - mean) / max


## rendering class

class Rendering():
    def __init__(
        self,
        image_size = 128,
        points_per_pixel = 10,
        point_radius = 8e-4,
        gamma = 1e-3,
        num_points = 10**6,
        smplx_model_path = None,
        rank = 0
    ):

        self.device = torch.device('cuda:%d' % rank if torch.cuda.is_available() else 'cpu')

        # rendering parameters
        self.image_size = image_size
        self.points_per_pixel = points_per_pixel
        self.point_radius = point_radius
        self.num_points = num_points

        # specific pulsar parameters
        self.gamma = gamma # gamma = 1.0 is mostly transparent : use values closer to 1 for geometry optimization
        self.znear = 1.0
        self.background = torch.ones((3,), device=self.device)

        # smplx model, faces and uvs are always the same
        self.smplx_model = smplx.SMPLXLayer(smplx_model_path, gender='neutral', device=self.device)
        self.smplx_faces = self.smplx_model.faces_tensor.unsqueeze(0)

        smplx_uv_path = os.path.join(smplx_model_path, 'smplx_uv.obj')
        self.smplx_uvs = self.get_smplx_uvs(smplx_uv_path)

        # TODO
        # camera view settings
        self.azimuths = np.arange(-180, 180, 1)
        self.elevations = np.arange(-30, 30, 1)
        elev_probs = np.exp( -np.square((self.elevations)/15) ) # exp( -(x/15)**2 )
        self.elev_probs = elev_probs / elev_probs.sum()

    def pulsar_renderer(self, cameras):
        raster_settings = PointsRasterizationSettings(
            image_size = self.image_size,
            radius = self.point_radius,
            points_per_pixel = self.points_per_pixel
        )

        rasterizer = PointsRasterizer(
            cameras = cameras,
            raster_settings = raster_settings
        )

        renderer = PulsarPointsRenderer(rasterizer=rasterizer).to(self.device)

        return renderer

    def get_mesh(self, mesh_obj_path, texture, batch_size):
        verts, faces, properties = load_obj(mesh_obj_path, load_textures=False)
        verts_uvs = properties.verts_uvs.unsqueeze(0).to(self.device)
        faces_uvs = faces.textures_idx.unsqueeze(0).to(self.device)
        verts_uvs = verts_uvs.repeat(batch_size, 1, 1) # shape = (b, V, 2)
        faces_uvs = faces_uvs.repeat(batch_size, 1, 1) # shape = (b, F, 3)

        # verts = normalize_verts(verts.unsqueeze(0))
        verts = verts.unsqueeze(0).to(self.device)
        faces = faces.verts_idx.unsqueeze(0).to(self.device)
        verts = verts.repeat(batch_size, 1, 1) # shape = (b, num_verts, 3)
        faces = faces.repeat(batch_size, 1, 1) # shape = (b, num_faces, 3)

        texture = torch.moveaxis(texture, 1, 3).clamp_(0., 1.)
        texture_uv = TexturesUV(maps=texture, faces_uvs=faces_uvs, verts_uvs=verts_uvs)

        return Meshes(verts, faces, texture_uv)

    def get_smplx_uvs(self, smplx_uv_path):
        obj_mesh = load_obj(smplx_uv_path, load_textures=False, device=self.device)
        faces_uvs = obj_mesh[1].textures_idx
        verts_uvs = obj_mesh[2].verts_uvs

        return faces_uvs, verts_uvs

    # camera rotation to mesh rotation
    # https://github.com/YadiraF/PIXIE/tree/master/pixielib/utils/util.py#L90
    def orth_project(self, x, camera):
        # x is of shape (b, num_verts, 3)
        camera = camera.clone().view(-1, 1, 3)
        x_trans = x[:,:,:2] + camera[:,:,1:]
        x_trans = torch.cat([x_trans, x[:,:,2:]], 2)
        transformed = (camera[:,:,:1] * x_trans)
        transformed[...,:2] = -transformed[...,:2]

        return transformed

    # smplx mesh from parameters provided via labels
    def get_smplx_mesh(self, labels, texture=None):
        global_orient, body_pose, jaw_pose, left_hand_pose, right_hand_pose, expression, betas, camera = labels
        b = global_orient.shape[0] # batch size

        # smplx vertices
        smplx_verts = self.smplx_model.forward(global_orient=global_orient,
                                                             body_pose=body_pose,
                                                             jaw_pose=jaw_pose,
                                                             left_hand_pose=left_hand_pose,
                                                             right_hand_pose=right_hand_pose,
                                                             expression=expression,
                                                             betas=betas, device=self.device)['vertices']

        # project vertices to image space
        smplx_verts = self.orth_project(smplx_verts, camera)

        # smplx texture
        if texture is None:
            color = torch.ones_like(smplx_verts, device=self.device)
            texture = TexturesVertex(color)
        else:
        faces_uvs, verts_uvs = self.smplx_uvs
            texturemap = torch.moveaxis(texture, 1, 3).clamp(0., 1.)
            texture = TexturesUV(texturemap, [faces_uvs] * b, [verts_uvs] * b)

        return Meshes(smplx_verts, [self.smplx_faces] * b, texture)

    # sample points from mesh
    def get_pointcloud(self, mesh):
        points_xyz, points_norms, points_text = sample_points_from_meshes(mesh, num_samples=self.num_points, return_normals=True, return_textures=True)

        if points_text.shape[2] > 3:
            points_xyz += points_norms * points_text[:, :, 3:4].tile([1, 1, 3])

        return Pointclouds(points=points_xyz, features=points_text[:, :, 0:3])

    # TODO
    # also, does not support greyscale nor transparent : make assert in model.py
    # rename label to param or smth that makes more sense
    def render(self, texture, label):
        batch_size = texture.shape[0]

        # azim = np.random.choice(self.azimuths, b, replace=True) if label is None else label[:,0]
        # elev = np.random.choice(self.elevations, b, replace=True, p=self.elev_probs) if label is None else label[:,1]
        # dist = 10

        # camera
        dist = 10
        azim = (180,) * batch_size
        elev = (0,) * batch_size
        R, T = look_at_view_transform(dist=10, elev=elev, azim=azim)
        cameras = PerspectiveCameras(focal_length=dist, R=R, T=T, device=self.device)
        renderer = self.pulsar_renderer(cameras)

        # get mesh and pointcloud
        mesh = self.get_smplx_mesh(label, texture)
        pointcloud = self.get_pointcloud(mesh)

        # pulsar rendering
        gamma = (self.gamma,) * batch_size
        znear = (self.znear,) * batch_size
        zfar = (2 * dist,) * batch_size
        image = renderer(pointcloud, gamma=gamma, znear=znear, zfar=zfar, bg_col=self.background)[..., :3]

        return torch.moveaxis(image, 3, 1)