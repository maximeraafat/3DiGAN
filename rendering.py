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
    AlphaCompositor,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PulsarPointsRenderer
)

## rendering class

class Rendering():
    def __init__(
        self,
        image_size = 128,
        points_per_pixel = 10,
        point_radius = 8e-4,
        gamma = 1e-3,
        num_points = 10**6,
        mesh_obj_path=None,
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

        self.smplx_model_path = smplx_model_path
        self.mesh_obj_path = mesh_obj_path

        # TODO : requirement, mesh_obj_path requires a dataset.json file with azimuth and elevations! focal length is fixed
        if smplx_model_path:
            # smplx model, faces and uvs are always the same
            self.smplx_model = smplx.SMPLXLayer(smplx_model_path, gender='neutral').to(self.device)
            self.smplx_faces = self.smplx_model.faces_tensor.unsqueeze(0)

            smplx_uv_path = os.path.join(smplx_model_path, 'smplx_uv.obj')
            self.smplx_uvs = self.get_smplx_uvs(smplx_uv_path)

        else: # mesh_obj_path is not None
            mesh_object = load_obj(mesh_obj_path, load_textures=False, device=self.device)
            self.obj_verts = self.normalize_verts( mesh_object[0].unsqueeze(0) )
            self.obj_faces = mesh_object[1].verts_idx # .unsqueeze(0)

            obj_faces_uvs = mesh_object[1].textures_idx
            obj_verts_uvs = mesh_object[2].verts_uvs
            self.obj_uvs = obj_faces_uvs, obj_verts_uvs

    def point_renderer(self, cameras):
        raster_settings = PointsRasterizationSettings(
            image_size = self.image_size,
            radius = self.point_radius,
            points_per_pixel = self.points_per_pixel
        )

        rasterizer = PointsRasterizer(
            cameras = cameras,
            raster_settings = raster_settings
        )

        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=self.background)
            ).to(self.device)

        return renderer

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

    # normalize vertices by centering and scaling
    def normalize_verts(self, verts):
        mean = verts.mean(axis=1) # vertices mean
        max = (verts - mean).square().sum(axis=2).sqrt().max() # largest variation from mean
        return (verts - mean) / max

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
        batch_size = global_orient.shape[0]

        # smplx vertices
        smplx_verts = self.smplx_model.forward(global_orient=global_orient,
                                                             body_pose=body_pose,
                                                             jaw_pose=jaw_pose,
                                                             left_hand_pose=left_hand_pose,
                                                             right_hand_pose=right_hand_pose,
                                                             expression=expression,
                                                             betas=betas)['vertices'].to(self.device)

        # project vertices to image space
        smplx_verts = self.orth_project(smplx_verts, camera)

        # smplx texture
        if texture is None:
            color = torch.ones_like(smplx_verts, device=self.device)
            texture = TexturesVertex(color)
        else:
            faces_uvs, verts_uvs = self.smplx_uvs
            texturemap = torch.moveaxis(texture, 1, 3).clamp(0., 1.)
            texture = TexturesUV(texturemap, [faces_uvs] * batch_size, [verts_uvs] * batch_size)

        return Meshes(smplx_verts, self.smplx_faces.repeat(batch_size, 1, 1), texture)

    def get_obj_mesh(self, batch_size, texture=None):
        if texture is None:
            color = torch.ones_like(self.obj_verts, device=self.device)
            texture = TexturesVertex(color)
        else:
            faces_uvs, verts_uvs = self.obj_uvs
            texturemap = torch.moveaxis(texture, 1, 3).clamp(0., 1.)
            texture = TexturesUV(texturemap, [faces_uvs] * batch_size, [verts_uvs] * batch_size)

        return Meshes(self.obj_verts.repeat(batch_size, 1, 1), self.obj_faces.repeat(batch_size, 1, 1), texture)

    # sample points from mesh
    def get_pointcloud(self, mesh):
        points_xyz, points_norms, points_text = sample_points_from_meshes(mesh, num_samples=self.num_points, return_normals=True, return_textures=True)

        if points_text.shape[2] > 3:
            points_xyz += points_norms * points_text[:, :, 3:4].tile([1, 1, 3])

        return Pointclouds(points=points_xyz, features=points_text[:, :, 0:3])

    # TODO
    # also, does not support greyscale nor transparent : make assert in model.py
    # rename label to param or smth that makes more sense
    def render(self, texture, label, renderer='pulsar'):
        assert renderer in ('default', 'pulsar'), 'renderer has to be default or pulsar'
        batch_size = texture.shape[0]

        # camera
        dist = 10
        azim = (180,) if self.smplx_model_path else label[:,0]
        elev = (0,) if self.smplx_model_path else label[:,1]
        azim *= batch_size
        elev *= batch_size

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = PerspectiveCameras(focal_length=dist, R=R, T=T, device=self.device)
        r = self.pulsar_renderer(cameras) if renderer == 'pulsar' else self.point_renderer(cameras)

        # get mesh and pointcloud
        mesh = self.get_smplx_mesh(label, texture) if self.smplx_model_path else self.get_obj_mesh(batch_size, texture)
        pointcloud = self.get_pointcloud(mesh)

        # pulsar rendering
        gamma = (self.gamma,) * batch_size
        znear = (self.znear,) * batch_size
        zfar = (2 * dist,) * batch_size
        image = r(pointcloud, gamma=gamma, znear=znear, zfar=zfar, bg_col=self.background)[..., :3]

        return torch.moveaxis(image, 3, 1)