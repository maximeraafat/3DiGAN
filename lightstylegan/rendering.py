import torch
import numpy as np
from typing import Union

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV

from pytorch3d.renderer import (
    look_at_view_transform,
    OrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
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
        transparent = False,
        rank = 0
    ):

        self.image_size = image_size
        self.sigma = sigma
        self.gamma = gamma
        self.faces_per_pixel = faces_per_pixel
        self.transparent = transparent

        self.background_color = (0., 0., 0.) if transparent else (1., 1., 1.)

        self.device = torch.device('cuda:%d' % rank if torch.cuda.is_available() else 'cpu')

        # mesh and smplx uv .obj paths
        self.mesh_obj_path = 'mesh.obj'
        self.smplx_uv_path = 'smplx/smplx_uv.obj'

        # camera view settings
        self.azimuths = np.arange(-180, 180, 1)
        self.elevations = np.arange(-30, 30, 1)
        elev_probs = np.exp( -np.square((self.elevations)/15) ) # exp( -(x/15)**2 )
        self.elev_probs = elev_probs / elev_probs.sum()

    def get_renderer(self, cameras, transparent=False):
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
        phong_renderer = MeshRenderer(
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

        return phong_renderer

    def get_mesh(self, mesh_obj_path, batch_size):
        mesh_object = load_obj(mesh_obj_path, load_textures=False)
        verts = normalize_verts( mesh_object[0].unsqueeze(0) )
        faces = mesh_object[1].verts_idx.unsqueeze(0)
        verts = verts.repeat(batch_size, 1, 1) # shape = (b, num_verts, 3)
        faces = faces.repeat(batch_size, 1, 1) # shape = (b, num_faces, 3)

        return Meshes(verts, faces).to(self.device)

    def get_smplx_uvs(self, smplx_uv_path, batch_size):
        smplx_object = load_obj(smplx_uv_path, load_textures=False)
        verts_uvs = smplx_object[2].verts_uvs.unsqueeze(0).to(self.device)
        faces_uvs = smplx_object[1].textures_idx.unsqueeze(0).to(self.device)
        verts_uvs = verts_uvs.repeat(batch_size, 1, 1) # shape = (b, V, 2)
        faces_uvs = faces_uvs.repeat(batch_size, 1, 1) # shape = (b, F, 3)

        return verts_uvs, faces_uvs

    def get_textured_mesh(self, mesh, uvs, texture):
        texture = torch.moveaxis(texture, 1, 3).clamp_(0., 1.)
        texture_uv = TexturesUV(maps=texture, faces_uvs=uvs[1], verts_uvs=uvs[0])

        return Meshes(mesh.verts_padded(), mesh.faces_padded(), texture_uv).to(self.device)

    def render(self, texture, supervision=False):
        b = texture.shape[0] # batch size

        uvs = self.get_smplx_uvs(self.smplx_uv_path, b)
        mesh = self.get_mesh(self.mesh_obj_path, b)
        textured_mesh = self.get_textured_mesh(mesh, uvs, texture)

        azim = np.random.choice(self.azimuths, b, replace=True) if not supervision else 90
        elev = np.random.choice(self.elevations, b, replace=True, p=self.elev_probs) if not supervision else 0
        dist = 3

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = OrthographicCameras(R=R, T=T, device=self.device)

        renderer = self.get_renderer(cameras)
        if self.transparent:
            image = renderer(textured_mesh)
        else:
            image = renderer(textured_mesh)[..., :3]

        return torch.moveaxis(image, 3, 1)
