import argparse
import os
import cv2
import torch
import smplx
import numpy as np
from PIL import Image

from neural_rendering import neural_renderer
from utils.download_humbi import download_subject, get_pose, remove_subject
from utils.smplx_to_disps import smplx2disps
from utils.inpainting import get_disps_inpaint

SUBJECT_IDS = 'range(1, 618)'

parser = argparse.ArgumentParser()

parser.add_argument('--subjects', default=SUBJECT_IDS,
                    type=str, help='list or range of subject ids as string')
parser.add_argument('--gdrive', metavar='PATH', default='', type=str,
                    help='path to google drive (if on colab)')
parser.add_argument('--iters', metavar='INT', default=30,
                    type=int, help='number of integers for neural rendering per camera')
parser.add_argument('--subdivision', action='store_true',
                    help='whether to apply subdivision to the smplx mesh')
parser.add_argument('--saveobj', action='store_true',
                    help='whether to store neural rendering progress in .obj file every 3 iterations per subject')
parser.add_argument('--smoothing', action='store_true',
                    help='whether to slightly smooth mesh after learning vertex displacements')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.manual_seed(0)

    subjects = eval(args.subjects)
    assert( isinstance(subjects, list) or isinstance(subjects, range) ), '--subjects needs to be a valid list or range passed as a string'

    subdivision = args.subdivision

    obj_path = args.gdrive + 'smplx/smplx_uv.obj'
    subd_obj_path = args.gdrive + 'smplx/subd_smplx_uv.obj'
    uv_mask_img = args.gdrive + 'smplx/smplx_uv.png'

    save_path_objs = None
    if args.saveobj:
        save_path_objs = args.gdrive + 'humbi_output/humbi_smplx_objs'

    save_path_imgs = args.gdrive + 'humbi_output/humbi_smplx_imgs'
    save_path_npz = args.gdrive + 'humbi_output/humbi_smplx_npz'

    smplx_model_path = args.gdrive + 'smplx'
    smplx_model = smplx.SMPLXLayer(smplx_model_path, 'neutral').to(device)

    smoothing = args.smoothing

    attributes = 'body'

    for subject in subjects:
        # Do not download subject data if it already exists
        exists = os.path.exists('subject_%d' % subject)
        if exists:
            loaded = True

        if not exists:
            loaded = download_subject(subject, [attributes])

        if loaded:
            pose = get_pose(subject, attributes)

            # Neural rendering
            if not subdivision:
                geometry, texture = neural_renderer(smplx_model, subject, pose, args.iters, obj_path, subdivision, rescale_factor=2, save_path=save_path_objs)
            else:
                geometry, texture = neural_renderer(smplx_model, subject, pose, args.iters, subd_obj_path, subdivision, rescale_factor=2, save_path=save_path_objs)

            # Extract geometry
            global_orient, transl, body_pose, left_hand_pose, right_hand_pose, jaw_pose, expression, betas, scale, verts_disps = geometry

            # Store geometry into displacements along normals + get displaced and initial mesh
            learned_geometry = smplx2disps(smplx_model, betas, scale, verts_disps, subdivision, smoothing=2)[0]

            # Construct displacement map by interpolating values between uv vertex coordinates (inpainting) : for now only available for no subdivision!
            if not subdivision:
                displacement_map = get_disps_inpaint(subject, learned_geometry, obj_path, uv_mask_img, mask_disps=True)[0]

            # Save rgb color map and displacement map as textures
            os.makedirs(save_path_imgs, exist_ok=True)
            rgb_filename = os.path.join(save_path_imgs, 'rgb_texture_%d.png' % subject)
            disp_filename = os.path.join(save_path_imgs, 'disp_texture_%d.tiff' % subject)

            nrm_rgb_map = (texture[0].cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(nrm_rgb_map).save(rgb_filename)
            if not subdivision:
                nrm_disps_map = cv2.normalize(displacement_map.numpy(), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                Image.fromarray(nrm_disps_map, mode='F').save(disp_filename)

            # Save geometry in npz file
            body_pose = body_pose.cpu().numpy()
            betas = betas.cpu().numpy()
            scale = scale.cpu().numpy()
            verts_disps = verts_disps.cpu().numpy()
            learned_geometry = learned_geometry.cpu().numpy()

            os.makedirs(save_path_npz, exist_ok=True)
            geometry_filename = os.path.join(save_path_npz, 'output_subject_%d.npz' % subject)
            np.savez(geometry_filename, body_pose=body_pose, betas=betas, scale=scale, verts_disps=verts_disps, learned_geometry=learned_geometry)

            '''
            # Loading npz file
            npzfile = np.load(outfile)
            npzfile.files # see all stored arrays in npz file
            npzfile['body_pose'] # call body_pose array stored into npz file
            '''

        # do not remove subject data if it already existed
        if not exists and loaded:
            remove_subject(subject)

    print('\nDone!')


if __name__ == '__main__':
    main()
