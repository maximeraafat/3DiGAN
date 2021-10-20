import argparse
import os
import shutil
import zipfile
import urllib.request as request
from PIL import Image
from tqdm import tqdm


ATTRIBUTES = ['body', 'body_texture', 'face', 'face_texture', 'gaze', 'gaze_texture', 'cloth', 'hand']
SUBJECT_IDS = range(1, 618)  # all subjects by default
MAPS = ['mean', 'median', 'std', 'var']
POSES = ['00000001'] # default pose
SAVE_PATH = 'humbi_maps'

parser = argparse.ArgumentParser()

parser.add_argument('--attributes', metavar='STR', default='body_texture',
                    type=str, nargs='+', help='list of body attributes')
parser.add_argument('--subjects', metavar='INT', default=SUBJECT_IDS,
                    type=int, nargs='+', help='list of subject ids')
parser.add_argument('--maps', metavar='STR', default=MAPS,
                    type=str, nargs='+', help='list of maps')
parser.add_argument('--save', metavar='PATH', default=SAVE_PATH, type=str,
                    help='path to which the extracted maps will be stored')
parser.add_argument('--gdrive', metavar='PATH', default='', type=str,
                    help='path to google drive (if running on colab): will store maps under gdrive/save')

args = parser.parse_args()


### Download humbi dataset
def download_humbi_maps(attributes = ATTRIBUTES, subject_ids = SUBJECT_IDS,
                        maps = MAPS, poses = POSES, save_path = SAVE_PATH,
                        root_url = 'https://humbi-dataset.s3.amazonaws.com',
                        target_dir = os.getcwd()):
    if not isinstance(attributes, list):
        attributes = [attributes]

    if isinstance(subject_ids, int):
        subject_ids = range(subject_ids, subject_ids + 1)

    if not isinstance(maps, list):
        maps = [attributes]

    if not isinstance(poses, list):
        poses = [poses]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for subject in tqdm(subject_ids):
        for attribute in attributes:
            loaded = load_part(attribute, subject, root_url, target_dir)
            if loaded:
                for map in maps:
                    for pose in poses:
                        image = extract_image(attribute, subject, map, pose, target_dir)
                        save_to_drive(image, save_path, attribute, subject, map)
                remove_subject(attribute, subject, target_dir)

    return


### Download and extract attribute from single subject of humbi dataset
# Return True if and only if download succeeded
def load_part(attribute, subject,
              root_url = 'https://humbi-dataset.s3.amazonaws.com',
              target_dir = os.getcwd()):
    part_name = attribute + '_subject'
    zip_file = 'subject_%d.zip' % subject # OR 'subject_' + str(subject) + '.zip'
    url = os.path.join(root_url, part_name, zip_file)
    target_path = os.path.join('%s_%d.zip' % (part_name, subject))

    try:
        request.urlretrieve(url, target_path) # OR !wget url
        downloaded_zip = zipfile.ZipFile(target_path)
        downloaded_zip.extractall(target_dir) # !unzip downloaded_zip
        return True
    except request.HTTPError:
        print(' %s attribute is missing for subject %d' % (attribute, subject))
        return False


### Extract map from a subject
# Return extracted image
def extract_image(attribute, subject, map, pose = '00000001', target_dir = os.getcwd()):
    assert(map in ['mean', 'median', 'std', 'var']), "map needs to be one of ['mean', 'median', 'std', 'var']"

    path_to_subject = os.path.join(target_dir, 'subject_%d' % subject)
    map_name = map + '_map' + '_hot' * (map == 'std' or map == 'var') + '.png'

    if attribute in ['body', 'body_texture', 'cloth']:
        attribute_path = 'body'
    elif attribute in ['face', 'face_texture']:
        attribute_path = 'face'
    elif attribute in ['gaze', 'gaze_texture']:
        attribute_path = 'gaze'
    else:
        attribute_path = attribute

    poses_path = os.path.join(path_to_subject, attribute_path)
    if pose not in os.listdir(poses_path):
        new_pose = sorted(os.listdir(poses_path))[0]
        print('\n pose %s not available for subject %d, we replace with first pose available : %s' % (pose, subject, new_pose))
        pose = new_pose

    image_path = os.path.join(path_to_subject, attribute_path, pose, 'appearance', map_name)
    image = Image.open(image_path)

    return image


### Store image to drive
def save_to_drive(image, save_path, attribute, subject, map):
    assert(map in ['mean', 'median', 'std', 'var']), "map needs to be one of ['mean', 'median', 'std', 'var']"

    outer_directory = 'humbi_' + attribute
    inner_directory = attribute + '_' + map + 's'
    filename = map + '_subject_%d.png' % subject

    store_path = os.path.join(save_path, outer_directory, inner_directory, filename)
    os.makedirs(os.path.dirname(store_path), exist_ok = True)

    image.save(store_path, 'PNG')

    return


### Clean disk
def remove_subject(attribute, subject, target_dir = os.getcwd()):
    subject_directory = os.path.join(target_dir, 'subject_%d' % subject)
    subject_zip = os.path.join(target_dir, attribute + '_subject_%d.zip' % subject)

    shutil.rmtree(subject_directory)
    os.remove(subject_zip)

    return



save_path = os.path.join(args.gdrive, args.save)
print('Subjects data will be downloaded to :', save_path)
download_humbi_maps(attributes = args.attributes, subject_ids = args.subjects,
                    maps = args.maps, save_path = save_path)
