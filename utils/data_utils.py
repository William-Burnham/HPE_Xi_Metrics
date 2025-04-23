import os
import json
import numpy as np
import requests

from alive_progress import alive_bar # type: ignore

from utils.kp_utils import project_3d_to_2d
from utils.mem_utils import count, dict_size

from metrics.angles import calculate_angles

def load_file_paths(directory):
    file_paths = os.listdir(directory)
    for i in range(len(file_paths)): file_paths[i] = os.path.join(directory, file_paths[i])
    return file_paths

def make_path(save_dir, file_path, subject, camera, model_type, extension):
    split_filename = os.path.splitext(os.path.basename(file_path))[0] # Get the filename without extension from file_path
    return f"{os.path.join(save_dir, model_type, subject, camera, split_filename)}.{extension}"

def download_file(url, save_path):
    # Download file if it does not exist.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {save_path}")
    else:
        print(f"{save_path} already exists, skipping download.")

def find_json_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_paths.append(os.path.join(root, file))
    return file_paths

def save_predictions(model_type, keypoint_dict, keypoints, angles, ang_vels, ang_accs, confidences, save_dir, file_path, subject, duration, camera):
    save_path = make_path(save_dir, file_path, subject, camera, model_type, "json")

    save_data = {'joint_labels': keypoint_dict, 'keypoints': keypoints, 'angles': angles, 'ang_vels': ang_vels, 'ang_accs': ang_accs, 'confidences': confidences, 'prediction_duration': duration}

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, mode='w') as file:
        json.dump(save_data, file, indent=4)

def save_metrics(model_type, metric_dict, save_dir, file_path, subject, camera):
    save_path = make_path(save_dir, file_path, subject, camera, model_type, "json")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode='w') as file:
        json.dump(metric_dict, file, indent=4)

# https://github.com/sminchisescu-research/imar_vision_datasets_tools
def read_cam_params(cam_path):
    with open(cam_path) as f:
        cam_params = json.load(f)
        for key1 in cam_params:
            for key2 in cam_params[key1]:
                cam_params[key1][key2] = np.array(cam_params[key1][key2]) 
    return cam_params

# Edited from https://github.com/sminchisescu-research/imar_vision_datasets_tools
def read_data(data_root, dataset_name, subset, subj_name, action_name, camera_name):
    vid_path = '%s/%s/%s/%s/videos/%s/%s.mp4' % (data_root, dataset_name, subset, subj_name, camera_name, action_name)
    cam_path = '%s/%s/%s/%s/camera_parameters/%s/%s.json' % (data_root, dataset_name, subset, subj_name, camera_name, action_name)
    j3d_path = '%s/%s/%s/%s/joints3d_25/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)

    cam_params = read_cam_params(cam_path)

    with open(j3d_path) as f:
        j3ds = np.array(json.load(f)['joints3d_25'])
    seq_len = j3ds.shape[-3]
    
    return j3ds, cam_params

def compute_ground_truth(config):

    # Keypoints dictionary for the ground truth dataset
    GT_KP_DICT = {
        'lower_spine': 0,
        'left_hip': 1,
        'left_knee': 2,
        'left_ankle': 3,
        'right_hip': 4,
        'right_knee': 5,
        'right_ankle': 6,
        'centre_spine': 7,
        'upper_spine': 8,
        'neck': 9,
        'head': 10,
        'left_shoulder': 11,
        'left_elbow': 12,
        'left_wrist': 13,
        'right_shoulder': 14,
        'right_elbow': 15,
        'right_wrist': 16,
        'left_foot': 17,
        'left_foot_index': 18,
        'right_foot': 19,
        'right_foot_index': 20,
        'left_wrist_point': 21,
        'left_hand': 22,
        'right_wrist_point': 23,
        'right_hand': 24
    }

    # Nested folder query
    # Layer 1: per subject
    subjects_dir = os.path.join(config['data_root'], config['dataset_name'], config['subset'])
    subject_ids = os.listdir(subjects_dir)

    for subject in subject_ids:
        # Layer 2: per camera
        cameras_dir = os.path.join(subjects_dir, subject, 'videos')
        camera_ids = os.listdir(cameras_dir)

        for camera in camera_ids:

            # Layer 3: per video / per exercise
            videos_dir = os.path.join(cameras_dir, camera)
            file_paths = load_file_paths(videos_dir)

            # Progress bar
            with alive_bar(len(file_paths), title=f"{subject}, {camera}", length=16) as gt_bar:
            
                for file_path in file_paths:
                    action_name = os.path.basename(file_path)[:-4]

                    j3ds, cam_params = read_data(
                        config['data_root'],
                        config['dataset_name'],
                        config['subset'],
                        subject,
                        action_name,
                        camera
                    )

                    # Initialise list of joint positions per frame
                    joint_frames = []

                    # Get 2D joint projection for each frame
                    for j3d_frame in j3ds:
                        j2d_camera = project_3d_to_2d(
                            j3d_frame,
                            cam_params['intrinsics_w_distortion'],
                            'w_distortion',
                            cam_params,
                            config['frame_w'],
                            config['frame_h']
                        )
                        joint_frames.append(np.ndarray.tolist(j2d_camera))

                    # Calculate angles and angular derivatives
                    gt_angles, gt_ang_vels, gt_ang_accs = calculate_angles(joint_frames, GT_KP_DICT, config['fps'])

                    ground_truth = {}
                    ground_truth['keypoints'] = joint_frames
                    ground_truth['angles'] = gt_angles
                    ground_truth['ang_vels'] = gt_ang_vels
                    ground_truth['ang_accs'] = gt_ang_accs

                    # Save ground truth data to json file
                    gt_path = make_path(config['gt_dir'], file_path, subject, camera, "ground_truth", "json")
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
                    with open(gt_path, mode='w') as file:
                        json.dump(ground_truth, file, indent=4)

                    gt_bar()
