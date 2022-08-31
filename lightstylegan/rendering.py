import torch
import numpy as np
from typing import Union

from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import TexturesUV

from pytorch3d.renderer import (
    look_at_view_transform,
    OrthographicCameras,
    PerspectiveCameras,
    RasterizationSettings,
    PointsRasterizationSettings,
    MeshRenderer,
    PointsRenderer,
    MeshRasterizer,
    PointsRasterizer,
    HardPhongShader,
    AlphaCompositor,
    AmbientLights,
    PointLights,
    Materials,
    BlendParams
)


## normalize vertices by centering and scaling

def normalize_verts(verts):
    mean = verts.mean(axis=1) # vertices mean
    max = (verts - mean).square().sum(axis=2).sqrt().max() # largest variation from mean
    return (verts - mean) / max


## rendering class

class Rendering():
    def __init__(
        self,
        image_size = 128,
        sigma = 1e-7,
        gamma = 1e-7,
        faces_per_pixel = 1,
        points_per_pixel = 30,
        point_radius = 0.002, # TODO : optimal for cow rendering = 0.005
        transparent = False,
        rank = 0
    ):

        self.image_size = image_size
        self.sigma = sigma
        self.gamma = gamma
        self.faces_per_pixel = faces_per_pixel
        self.points_per_pixel = points_per_pixel
        self.point_radius = point_radius
        self.transparent = transparent

        self.background_color = (0., 0., 0.) if transparent else (1., 1., 1.)

        self.device = torch.device('cuda:%d' % rank if torch.cuda.is_available() else 'cpu')

        # mesh and smplx uv .obj paths
        self.mesh_obj_path = 'mesh.obj'
        self.smplx_uv_path = 'mesh.obj'

        # camera view settings
        self.azimuths = np.arange(-180, 180, 1)
        self.elevations = np.arange(-30, 30, 1)
        elev_probs = np.exp( -np.square((self.elevations)/15) ) # exp( -(x/15)**2 )
        self.elev_probs = elev_probs / elev_probs.sum()

    def mesh_renderer(self, cameras, transparent=False):
        blend_params = BlendParams(background_color=self.background_color, sigma=self.sigma * transparent, gamma=self.gamma) # settings for opacity and sharpness of edges
        materials = Materials(device=self.device, specular_color=[[0., 0., 0.]], shininess=0.)
        # lights = AmbientLights(device=self.device)
        lights = PointLights(device=self.device, location=((2., 2., 2.),))

        # settings for rasterization and shading
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=np.log(1./1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=self.faces_per_pixel,
            perspective_correct=False
        )

        # create phong renderer by composing a rasterizer and a shader
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                materials=materials,
                blend_params=blend_params
            )
        )

        return renderer

    # See SMPL-A renderer : https://gist.github.com/sergeyprokudin/2122726cdadb18e91e9d425c7514733d
    def point_renderer(self, cameras):
        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=self.point_radius,
            points_per_pixel=self.points_per_pixel
        )

        rasterizer = PointsRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=self.background_color)
        )

        return renderer

    def get_mesh(self, mesh_obj_path, batch_size):
        verts, faces, properties = load_obj(mesh_obj_path, load_textures=False)
        # verts = normalize_verts( verts.unsqueeze(0) )
        verts = verts.unsqueeze(0)
        faces = faces.verts_idx.unsqueeze(0)
        verts = verts.repeat(batch_size, 1, 1) # shape = (b, num_verts, 3)
        faces = faces.repeat(batch_size, 1, 1) # shape = (b, num_faces, 3)

        return Meshes(verts, faces).to(self.device)

    def get_smplx_uvs(self, smplx_uv_path, batch_size):
        verts, faces, properties = load_obj(smplx_uv_path, load_textures=False)
        verts_uvs = properties.verts_uvs.unsqueeze(0).to(self.device)
        faces_uvs = faces.textures_idx.unsqueeze(0).to(self.device)
        verts_uvs = verts_uvs.repeat(batch_size, 1, 1) # shape = (b, V, 2)
        faces_uvs = faces_uvs.repeat(batch_size, 1, 1) # shape = (b, F, 3)

        return verts_uvs, faces_uvs

    def get_textured_mesh(self, mesh, uvs, texture):
        texture = torch.moveaxis(texture, 1, 3).clamp_(0., 1.)
        texture_uv = TexturesUV(maps=texture, faces_uvs=uvs[1], verts_uvs=uvs[0])

        return Meshes(mesh.verts_padded(), mesh.faces_padded(), texture_uv).to(self.device)

    # See SMPL-A forward : https://gist.github.com/sergeyprokudin/d9c27822ceccff8de9830fb09202d7cf
    def get_pointcloud(self, mesh, num_samples=10**5): # TODO : optimal for cow rendering=10**6
        points_xyz, points_norms, points_text = sample_points_from_meshes(mesh, num_samples=num_samples, return_normals=True, return_textures=True)

        if points_text.shape[2] > 3:
            points_xyz += points_norms * points_text[:,:,3:4].tile([1, 1, 3])

        return Pointclouds(points=points_xyz, features=points_text[:,:,0:3])

    def render(self, texture, label=None):
        b = texture.shape[0] # batch size

        uvs = self.get_smplx_uvs(self.smplx_uv_path, b)
        mesh = self.get_mesh(self.mesh_obj_path, b)
        textured_mesh = self.get_textured_mesh(mesh, uvs, texture)

        azim = 180 # np.random.choice(self.azimuths, b, replace=True) if label is None else label[:,0]
        elev = 0 # np.random.choice(self.elevations, b, replace=True, p=self.elev_probs) if label is None else label[:,1]
        dist = 10 # 3

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        # cameras = OrthographicCameras(R=R, T=T, device=self.device)
        cameras = PerspectiveCameras(focal_length=dist, R=R, T=T, device=self.device)

        renderer = self.point_renderer(cameras)
        textured_pcl = self.get_pointcloud(textured_mesh)

        image = renderer(textured_pcl) if self.transparent else renderer(textured_pcl)[..., :3]

        return torch.moveaxis(image, 3, 1)
