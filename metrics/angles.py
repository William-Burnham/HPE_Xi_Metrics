import numpy as np
from scipy import signal, ndimage
import math

import matplotlib.pyplot as plt

from typing import Tuple

from models.hpe_models.hpe_model import HPE_Model

def angle_from_points(f_keypoints: list, kp_dict: dict, select_joints: list[str]) -> float:
    """
    Args:
        f_keypoints (list): List of skeleton keypoints for a single frame in format [x1, y1, c1, ... , xn, yn, cn]
        select_joints (list): List of three joint labels to select from keypoint dictionary
        kp_dict (dict): Keypoint dictonary of the current model
    Returns:
        (float): Calculated angle in radians
    """

    # Get keypoint indices
    joint_indices = [kp_dict.get(joint) for joint in select_joints]

    # Extract coordinates
    joint_d = np.array([f_keypoints[joint_indices[0]][0], f_keypoints[joint_indices[0]][1]])
    joint_e = np.array([f_keypoints[joint_indices[1]][0], f_keypoints[joint_indices[1]][1]])
    joint_f = np.array([f_keypoints[joint_indices[2]][0], f_keypoints[joint_indices[2]][1]])

    # Calculate vectors
    vector_A = joint_d - joint_e  # Vector DE
    vector_B = joint_f - joint_e  # Vector FE

    # Compute signed angle using arctan2
    angle = np.arctan2(np.cross(vector_A, vector_B), np.dot(vector_A, vector_B))

    return angle

def calculate_angles(keypoints: list, kp_dict: dict, fps: int) -> Tuple[dict, dict, dict]:

    # This dictionary defines the lables of joints needed for each important angle
    JOINT_CONNECTIONS: dict = {
        '0': ['left_knee', 'left_ankle', 'left_foot_index'],
        '1': ['right_knee', 'right_ankle', 'right_foot_index'],
        '2': ['left_hip', 'left_knee', 'left_ankle'],
        '3': ['right_hip', 'right_knee', 'right_ankle'],
        '4': ['right_hip', 'left_hip', 'left_knee'],
        '5': ['left_hip', 'right_hip', 'right_knee'],
        '6': ['left_shoulder', 'left_hip', 'left_knee'],
        '7': ['right_shoulder', 'right_hip', 'right_knee'],
        '8': ['left_hip', 'left_shoulder', 'left_elbow'],
        '9': ['right_hip', 'right_shoulder', 'right_elbow'],
        '10': ['right_shoulder', 'left_shoulder', 'left_elbow'],
        '11': ['left_shoulder', 'right_shoulder', 'right_elbow'],
        '12': ['left_shoulder', 'left_elbow', 'left_wrist'],
        '13': ['right_shoulder', 'right_elbow', 'right_wrist']
    }

    # Initialise dictionaries to store angles, angular velocities, and angular accelerations
    angles: dict = {}
    ang_vels: dict = {}
    ang_accs: dict = {}

    # Calculate angles for each selected joint (theta)
    for theta, joints in JOINT_CONNECTIONS.items():

        # Ensure all joints exist in kp_dict
        if any(j not in kp_dict for j in joints):
            missing = [j for j in joints if j not in kp_dict]
            print(f"Skipping {theta=}: Missing keypoints {missing}")
            angles[theta] = None
            continue

        # Calculate angles for every frame
        current_angles: list = []
        for kpi, f_keypoints in enumerate(keypoints):
            predicted_angle = float(angle_from_points(f_keypoints, kp_dict, joints))
            current_angles.append(float(angle_from_points(f_keypoints, kp_dict, joints)))
        angles[theta] = current_angles

        # Calculate angular velocities and angular accelerations using a butterworth filter to reduce noise
        ang_vels[theta], _ = butterworth_derivative(current_angles, fps, is_angles=True)
        ang_accs[theta], _ = butterworth_derivative(ang_vels[theta], fps, is_angles=False)

    return angles, ang_vels, ang_accs

# From two angles (0 to 2pi) calculate the smallest signed difference between them
def angular_difference(a, b):
    """
    Args:
        a (float): Source angle (-pi to +pi)
        b (float): Target angle (-pi to +pi)
    Returns:
        (float): Smallest signed difference
    """

    difference = b - a
    # Compensate for wrap-around from -pi to +pi
    if difference > np.pi: difference -= 2 * np.pi
    if difference < -np.pi: difference += 2 * np.pi

    return difference

def derivative(input: list, fps: int, nan_indices: list = []) -> list:

    deriv = []
    for i in range(1, len(input) - 1):
        if any(index in nan_indices for index in [i-1, i, i+1]):
            # Append nan if there is a nan in any position between i-1 and i+1
            deriv.append(np.nan)
        else:
            deriv.append(float((input[i+1] - input[i-1]) * (fps / 2)))

    # First index
    if any(index in nan_indices for index in [0, 1]):
        deriv.insert(0, np.nan)
    else:
        deriv.insert(0, float((input[1] - input[0]) * (fps)))

    # Last index
    if any(index in nan_indices for index in [len(input)-2, len(input)-1]):
        deriv.append(np.nan)
    else:
        deriv.append(float((input[-1] - input[-2]) * (fps)))

    return deriv

# Not used - butterworth is more common for clinical practice
def gaussian_derivative(input: list, sigma: float = 1) -> list:

    filtered = ndimage.gaussian_filter(input, sigma)
    derived = derivative(filtered)

    return derived

# Set all nan values to the value immediately before them and return indices of all discovered nans
def handle_nans(input: list) -> Tuple[list, list]:

    no_nans = input.copy()
    nan_indices = []
    for index in range(len(no_nans)):
        if np.isnan(no_nans[index]):
            if index == 0:
                # If first in the sequence, set to zero
                no_nans[index] = 0.0
            else:
                no_nans[index] = no_nans[index-1]
            nan_indices.append(index)

    return no_nans, nan_indices

def remove_jump_discontinuity(jd_list: list) -> Tuple[list, list, list]:

    # Searches through list and adds or subtracts following angles by 2pi (tau) at each jump discontinuity ({x[i]>0.5pi and x[i+1]<-0.5pi} or vice versa)

    jd_list = np.array(jd_list)

    displacement = 0 # Track displaced to factor into comparison
    u_bound = []
    l_bound = []

    for i in range(len(jd_list)-1):
        u_bound.append(displacement+np.pi)
        l_bound.append(displacement-np.pi)
        if jd_list[i]>(0.5*np.pi)+(displacement) and jd_list[i+1]<(-0.5*np.pi)+(displacement):
            jd_list[(i+1):]+=2*np.pi
            displacement+=2*np.pi
        elif jd_list[i]<(-0.5*np.pi)+(displacement) and jd_list[i+1]>(0.5*np.pi)+(displacement):
            jd_list[(i+1):]-=2*np.pi
            displacement-=2*np.pi

    u_bound.append(displacement+np.pi)
    l_bound.append(displacement-np.pi)

    return jd_list, u_bound, l_bound

#https://help.vicon.com/space/Nexus216/11614296/Pipeline+tools:
"""
Filter subject model outputs using a low-pass digital Butterworth filter.
The filter is by default setup as recommended in Winter, D.A. Biomechanics of Motor Control
and Human Movement to filter out signal noise above 6 Hz using a Fourth Order filter with zero lag.
"""
def butterworth_derivative(input: list, fps: int, order: int = 4, cutoff: float = 6, is_angles: bool = False) -> list:
    """
        args:
            input (list): List of unfiltered values.
            fps (int): Frames per second.
            order (int): Order for Butterworth filter.
            cutoff (float): Cutoff for Butterworth filter.
            is_angles (bool): When deriving from angles set to True, this will account for -pi to +pi wrapping in angles.
        returns:
            (list): Gradients of filtered values.
    """

    # Set NaN values to previous values, otherwise butterworth filter raises error, the affected derivative indexes are returned to carry through non-prediction
    input_no_nans, nan_indices = handle_nans(input)

    # Remove jump discontinuity for -pi to pi
    if is_angles:

        input_no_nans, u_bound, l_bound = remove_jump_discontinuity(input_no_nans)

    # Get the filter coefficients
    b, a = signal.butter(order, cutoff, fs=fps, btype='low', analog=False)
    filtered = signal.filtfilt(b, a, input_no_nans)
    derived = derivative(filtered, fps, nan_indices)

    return derived, filtered