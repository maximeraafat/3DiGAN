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
        faces_per_pixel = 1,
        points_per_pixel = 10,
        point_radius = 0.0002,
        smplx_model_path = None,
        smplx_uv_path = None,
        rank = 0
    ):

        self.device = torch.device('cuda:%d' % rank if torch.cuda.is_available() else 'cpu')

        self.image_size = image_size
        self.faces_per_pixel = faces_per_pixel
        self.points_per_pixel = points_per_pixel
        self.point_radius = point_radius

        # mesh and smplx uv .obj paths
        self.mesh_obj_path = 'mesh.obj'
        self.smplx_uv_path = 'mesh.obj'

        # smplx model, faces and uvs are always the same
        self.smplx_model = smplx.SMPLXLayer(smplx_model_path, gender='neutral').to(self.device)
        self.smplx_faces = self.smplx_model.faces_tensor.unsqueeze(0).to(self.device)
        self.smplx_uvs = self.get_smplx_uvs(smplx_uv_path)

        # camera view settings
        self.azimuths = np.arange(-180, 180, 1)
        self.elevations = np.arange(-30, 30, 1)
        elev_probs = np.exp( -np.square((self.elevations)/15) ) # exp( -(x/15)**2 )
        self.elev_probs = elev_probs / elev_probs.sum()


    def pulsar_renderer(self, cameras, point_radius, resolution):
        raster_settings = PointsRasterizationSettings(
            image_size = resolution,
            radius = point_radius,
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
        obj_mesh = load_obj(smplx_uv_path, load_textures=False)
        faces_uvs = obj_mesh[1].textures_idx.unsqueeze(0).to(self.device)
        verts_uvs = obj_mesh[2].verts_uvs.unsqueeze(0).to(self.device)

        return faces_uvs, verts_uvs

    # camera rotation to mesh rotation
    # see https://github.com/YadiraF/PIXIE/blob/fb493f5f0428ee32adef128eb6bc995069dd71e9/pixielib/utils/util.py#L90
    def orth_project(self, X, camera):
        # X is N x num_verts x 3
        camera = camera.clone().view(-1, 1, 3)
        X_trans = X[:,:,:2] + camera[:,:,1:]
        X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
        transformed = (camera[:,:,:1] * X_trans)
        transformed[...,:2] = -transformed[...,:2]

        return transformed

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
                                                             betas=betas)['vertices'].to(self.device)

        # project vertices to image space
        smplx_verts = self.orth_project(smplx_verts, camera)

        if texture is None:
            texture = TexturesVertex(torch.ones_like(smplx_verts))
            return Meshes(smplx_verts, self.smplx_faces.repeat(b, 1, 1), texture)

        # smplx uv coordinates
        faces_uvs, verts_uvs = self.smplx_uvs
        faces_uvs, verts_uvs = faces_uvs.repeat(b, 1, 1), verts_uvs.repeat(b, 1, 1)

        # smplx texture
        texture = torch.moveaxis(texture, 1, 3).clamp(0., 1.)
        texture_uv = TexturesUV(maps=texture, faces_uvs=faces_uvs, verts_uvs=verts_uvs)

        return Meshes(smplx_verts, self.smplx_faces.repeat(b, 1, 1), texture_uv)

    # sample points from mesh
    def get_pointcloud(self, mesh, num_samples=10**6):
        points_xyz, points_norms, points_text = sample_points_from_meshes(mesh, num_samples=num_samples, return_normals=True, return_textures=True)
        if points_text.shape[2] > 3:
            points_xyz += points_norms * points_text[:,:,3:4].tile([1, 1, 3])

        return Pointclouds(points=points_xyz, features=points_text[:,:,0:3])

    # TODO
    # gamma = 1.0 : mostly transparent
    # use higher gamma values (closer to one) and larger sphere sizes for geometry optimisation
    # also, does not support greyscale nor transparent : make assert in lightweight_gan.py
    # gamma, radius and num_samples to init and not as function arguments
    # rename label to param or smth that makes more sense
    def render(self, texture, label, gamma=1e-5, radius=0.0005, num_samples=10**6):
        batch_size = texture.shape[0]

        dist = 10
        azim = (180,) * batch_size
        elev = (0,) * batch_size

        # azim = np.random.choice(self.azimuths, b, replace=True) if label is None else label[:,0]
        # elev = np.random.choice(self.elevations, b, replace=True, p=self.elev_probs) if label is None else label[:,1]
        # dist = 10

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = PerspectiveCameras(focal_length=dist, R=R, T=T, device=self.device)

        # TODO : initialize renderer once
        renderer = self.pulsar_renderer(cameras, radius, resolution=self.image_size)

        # mesh = self.get_mesh(self.mesh_obj_path, texture, b)
        mesh = self.get_smplx_mesh(label, texture)
        pointcloud = self.get_pointcloud(mesh, num_samples)

        # pulsar rendering
        gamma = (gamma,) * batch_size
        znear =(1.0,) * batch_size
        zfar = (2 * dist,) * batch_size
        background = torch.ones((3,), device=self.device)
        image = renderer(pointcloud, gamma=gamma, znear=znear, zfar=zfar, bg_col=background)[..., :3]

        return torch.moveaxis(image, 3, 1)