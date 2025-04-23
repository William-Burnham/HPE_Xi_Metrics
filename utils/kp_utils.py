import numpy as np
import os
import json

from metrics.angles import calculate_angles

def reposition_keypoints(keypoints, confidences, swap_xy=False, flip_x=False, flip_y=False, confidence_factor=1, deconf_zeros=True):
    """
    args:
        keypoints: list of keypoints for each frame
        swap_xy: swap the x and y values of each keypoint
        flip_x: flip x positions over the centre
        flip_y: flip y positions over the centre
        confidence_factor: factor to multiply all confidence levels by
        deconf_zeros: for all positions predicted at (0.0, 0.0), set confidence to 0.0
    returns:
        (list, list): Keypoints and confidences
    """
    
    # Rearrange points to correct placements
    for keypoint_i, keypoint_value in enumerate(keypoints):

        for joint_i in range(0, len(keypoint_value)):

            if swap_xy:
                temp = keypoint_value[joint_i][0]
                keypoint_value[joint_i][0] = keypoint_value[joint_i][1]
                keypoint_value[joint_i][1] = temp
            
            if flip_x:
                keypoint_value[joint_i][0] = 1 - keypoint_value[joint_i][0]

            if flip_y:
                keypoint_value[joint_i][1] = 1 - keypoint_value[joint_i][1]

            if deconf_zeros and keypoint_value[joint_i][0] == 0 and keypoint_value[joint_i][0] == 0:
                confidences[keypoint_i][joint_i] = 0

            confidences[keypoint_i][joint_i] *= confidence_factor
        
        keypoints[keypoint_i] = keypoint_value

    return keypoints, confidences

# Edited from https://github.com/sminchisescu-research/imar_vision_datasets_tools
def project_3d_to_2d(j3d, intrinsics, intrinsics_type, cam_params, frame_w, frame_h):
    # 3D-2D projection
    j3d_in_camera = np.matmul(np.array(j3d) - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R']))
    if intrinsics_type == 'w_distortion':
        p = intrinsics['p'][:, [1, 0]]
        x = j3d_in_camera[:, :2] / j3d_in_camera[:, 2:3]
        r2 = np.sum(x**2, axis=1)
        radial = 1 + np.transpose(np.matmul(intrinsics['k'], np.array([r2, r2**2, r2**3])))
        tan = np.matmul(x, np.transpose(p))
        xx = x*(tan + radial) + r2[:, np.newaxis] * p
        proj = intrinsics['f'] * xx + intrinsics['c']
    elif intrinsics_type == 'wo_distortion':
        xx = j3d_in_camera[:, :2] / j3d_in_camera[:, 2:3]
        proj = intrinsics['f'] * xx + intrinsics['c']

    # Normalise between 0-1
    proj[:, 1] = (proj[:, 1] * -1 + frame_h) / frame_h 
    proj[:, 0] = proj[:, 0] / frame_w

    return proj