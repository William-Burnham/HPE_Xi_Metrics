import json
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

from utils.data_utils import make_path
from metrics.angles import angular_difference

from dtaidistance import dtw # type: ignore


def icc2_1(ground_truth, predictions):
    """
    Compute the Intraclass Correlation Coefficient (ICC2,1) for absolute agreement
    using a two-way random-effects model, considering only non-NaN values.

    Parameters:
        ground_truth (numpy.ndarray): 1D array of true values (e.g., Vicon).
        predictions (numpy.ndarray): 1D array of predicted values (e.g., HPE model output).

    Returns:
        float: ICC(2,1) value.
    """
    # Ensure inputs are numpy arrays
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

    # Mask NaN values
    valid_mask = ~np.isnan(ground_truth) & ~np.isnan(predictions)
    ground_truth = ground_truth[valid_mask]
    predictions = predictions[valid_mask]

    if len(ground_truth) == 0 or len(predictions) == 0:
        print("No valid data points available after removing NaNs.")
        return 0

    # Stack as a 2D array: Rows = subjects, Columns = raters (Vicon, HPE)
    data = np.vstack((ground_truth, predictions)).T  

    # Number of subjects (n) and raters (k=2 since we have GT & prediction)
    n, k = data.shape  

    # Compute mean squares
    grand_mean = np.mean(data)
    subject_means = np.mean(data, axis=1, keepdims=True)
    rater_means = np.mean(data, axis=0, keepdims=True)

    ss_total = np.sum((data - grand_mean) ** 2)  # Total sum of squares
    ss_subject = k * np.sum((subject_means - grand_mean) ** 2)  # Between-subject sum of squares
    ss_rater = n * np.sum((rater_means - grand_mean) ** 2)  # Between-rater sum of squares
    ss_error = ss_total - ss_subject - ss_rater  # Error sum of squares

    ms_subject = ss_subject / (n - 1)  # Mean square for subjects
    ms_error = ss_error / ((n - 1) * (k - 1))  # Mean square error

    # ICC(2,1) formula
    icc2_1_value = (ms_subject - ms_error) / (ms_subject + (k - 1) * ms_error)

    return icc2_1_value

def frechet_distance(X, Y):
    """
    Compute the discrete Fréchet distance between two time series X and Y.
    
        args:
            X (list[float]): Predicted list
            Y (list[float]): Ground truth list
        returns:
            (float): Fréchet distance
    """

    if len(X) != len(Y): raise IndexError("X and Y must have the same length")
    
    # Reshape to 2D
    timesteps = np.arange(len(X)).reshape(-1, 1)  # Generate time indices
    X = np.hstack((timesteps, np.array(X).reshape(-1, 1)))
    Y = np.hstack((timesteps, np.array(Y).reshape(-1, 1)))

    n, m = len(X), len(Y)
    D = cdist(X, Y, metric='euclidean')  # Compute pairwise distances
    L = np.zeros((n, m))
    
    L[0, 0] = D[0, 0]
    for i in range(1, n):
        L[i, 0] = max(L[i-1, 0], D[i, 0])
    for j in range(1, m):
        L[0, j] = max(L[0, j-1], D[0, j])
    
    for i in range(1, n):
        for j in range(1, m):
            L[i, j] = max(min(L[i-1, j], L[i, j-1], L[i-1, j-1]), D[i, j])
    
    return L[-1, -1]  # Final value is the Frechet distance

def z_score_normalise(input_list, mean: float, std: float) -> np.array:
    """
    Calculates the z-score normalisation of a list with equation:
    n = (x - mu) / sigma.

        args:
            input_list (list[float]): Sequence (x) to be normalised.
            mean (float): Mean (mu).
            std (float): Standard deviation (sigma).
        returns:
            (list[float]): Normalised sequence (n). 
    """
    x_array = np.array(input_list)
    norm_array = (x_array - mean) / std
    return norm_array

def dtw_distance(predicted, ground_truth):
    """
    Computes the Dynamic Time Warping (DTW) distance between predicted and ground truth signals.
    
    args:
        predicted (array-like): Predicted time series (angles, velocities, or accelerations)
        ground_truth (array-like): Ground truth time series.
    
    returns:
        (float): The DTW distance between the two sequences.
    """
    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)
    
    dtw_distance = dtw.distance(predicted, ground_truth)
    
    return dtw_distance

def p_r_f1_score(errors, threshold):

    tp = sum(1 for v in errors if isinstance(v, (int, float)) and v <= threshold)  # True Positive
    fp = sum(1 for v in errors if isinstance(v, (int, float)) and v > threshold)   # False Positive
    fn = sum(1 for v in errors if isinstance(v, float) and np.isnan(v))            # False Negative

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    return precision, recall, f1

def load_gt(file_path, subject, camera, config):
    gt_path = make_path(config['gt_dir'], file_path, subject, camera, "ground_truth", "json")
    with open(gt_path) as file:
        ground_truth = json.load(file)
    return ground_truth

def compare_to_gt(model, keypoints, angles, ang_vels, ang_accs, file_path, subject, camera, config):

    # Load the respective ground truth file
    ground_truth = load_gt(file_path, subject, camera, config)
    # dict_keys(['keypoints', 'angles', 'ang_vels', 'ang_accs'])

    # If there are more HPE-predicted frames than ground truth frames, remove appropriate
    # amount from the end of the sequence. Done in accordance with the code in:
    # https://github.com/sminchisescu-research/imar_vision_datasets_tools
    if len(ground_truth['keypoints']) < len(keypoints):
        frame_difference = len(keypoints) - len(ground_truth['keypoints'])
        # Remove same amount from all sequences, as they have the same number of frames
        keypoints = keypoints[:-frame_difference]
        for theta in ang_vels.keys():
            angles[theta] = angles[theta][:-frame_difference]
            ang_vels[theta] = ang_vels[theta][:-frame_difference]
            ang_accs[theta] = ang_accs[theta][:-frame_difference]

    # Define threshold constants
    ANGLE_THR_T = 0.0925
    ANGLE_THR_L = 0.186

    ANG_VELS_THR_T = 0.35
    ANG_VELS_THR_L = 0.571

    ANG_ACCS_THR_T = 1.833
    ANG_ACCS_THR_L = 3.491

    # Initialise memory
    angle_errors: dict = {}
    angle_error_means: list = []
    angle_error_medians: list = []
    angle_precisions: dict = {'tight': [], 'loose': []}
    angle_recalls: dict = {'tight': [], 'loose': []}
    angle_f1s: dict = {'tight': [], 'loose': []}
    #angle_dtws: list = []
    #angle_fds: list = []

    ang_vels_errors: dict = {}
    ang_vels_error_means: list = []
    ang_vels_error_medians: list = []
    ang_vels_precisions: dict = {'tight': [], 'loose': []}
    ang_vels_recalls: dict = {'tight': [], 'loose': []}
    ang_vels_f1s: dict = {'tight': [], 'loose': []}
    #ang_vels_dtws: list = []
    #ang_vels_fds: list = []

    ang_accs_errors: dict = {}
    ang_accs_error_means: list = []
    ang_accs_error_medians: list = []
    ang_accs_precisions: dict = {'tight': [], 'loose': []}
    ang_accs_recalls: dict = {'tight': [], 'loose': []}
    ang_accs_f1s: dict = {'tight': [], 'loose': []}
    #ang_accs_dtws: list =[]
    #ang_accs_fds:list = []

    for theta in ground_truth['angles'].keys():

        if config['do_ankles'] == False and theta in ['0','1']:
            continue

        if config['do_transverse_angles'] == False and theta in ['4', '5', '10', '11']:
            continue

        # If no angles were predicted, there are also no vels or accs. Skip
        if angles[theta] == None:
            continue

        # ANGLE ERRORS
        # --------------------------------------------------- #

        gt_theta_angles = ground_truth['angles'][theta]

        angle_errors[theta] = {}

        angle_errors_theta = np.abs(np.asarray([angular_difference(angles[theta][i], gt_theta_angles[i]) for i in range(len(angles[theta]))]))
        #angle_errors_theta = np.abs(np.asarray(gt_theta_angles) - np.asarray(angles[theta]))
        angle_errors[theta]['framewise_error'] = np.ndarray.tolist(angle_errors_theta)
        angle_errors[theta]['error_mean'] = np.nanmean(angle_errors_theta)
        angle_errors[theta]['error_median'] = np.nanmedian(angle_errors_theta)

        angle_error_means.append(angle_errors[theta]['error_mean'])
        angle_error_medians.append(angle_errors[theta]['error_median'])

        angle_icc = icc2_1(gt_theta_angles, angles[theta])
        angle_errors[theta]['ICC'] = angle_icc

        angle_errors[theta]['precision'] = {}
        angle_errors[theta]['recall'] = {}
        angle_errors[theta]['f1'] = {}

        p, r, f1 = p_r_f1_score(angle_errors_theta, ANGLE_THR_T)
        angle_errors[theta]['precision']['tight'] = p
        angle_errors[theta]['recall']['tight'] = r
        angle_errors[theta]['f1']['tight'] = f1
        angle_precisions['tight'].append(p)
        angle_recalls['tight'].append(r)
        angle_f1s['tight'].append(f1)

        p, r, f1 = p_r_f1_score(angle_errors_theta, ANGLE_THR_L)
        angle_errors[theta]['precision']['loose'] = p
        angle_errors[theta]['recall']['loose'] = r
        angle_errors[theta]['f1']['loose'] = f1
        angle_precisions['loose'].append(p)
        angle_recalls['loose'].append(r)
        angle_f1s['loose'].append(f1)

        # ANGULAR VELOCITY ERRORS
        # --------------------------------------------------- #

        gt_theta_ang_vels = ground_truth['ang_vels'][theta]

        ang_vels_errors[theta] = {}

        ang_vels_errors_theta = np.abs(np.asarray(gt_theta_ang_vels) - np.asarray(ang_vels[theta]))
        ang_vels_errors[theta]['framewise_error'] = np.ndarray.tolist(ang_vels_errors_theta)
        ang_vels_errors[theta]['error_mean'] = np.nanmean(ang_vels_errors_theta)
        ang_vels_errors[theta]['error_median'] = np.nanmedian(ang_vels_errors_theta)

        ang_vels_error_means.append(ang_vels_errors[theta]['error_mean'])
        ang_vels_error_medians.append(ang_vels_errors[theta]['error_median'])

        ang_vels_icc = icc2_1(gt_theta_ang_vels, ang_vels[theta])
        ang_vels_errors[theta]['ICC'] = ang_vels_icc

        ang_vels_errors[theta]['precision'] = {}
        ang_vels_errors[theta]['recall'] = {}
        ang_vels_errors[theta]['f1'] = {}

        p, r, f1 = p_r_f1_score(ang_vels_errors_theta, ANG_VELS_THR_T)
        ang_vels_errors[theta]['precision']['tight'] = p
        ang_vels_errors[theta]['recall']['tight'] = r
        ang_vels_errors[theta]['f1']['tight'] = f1
        ang_vels_precisions['tight'].append(p)
        ang_vels_recalls['tight'].append(r)
        ang_vels_f1s['tight'].append(f1)

        p, r, f1 = p_r_f1_score(ang_vels_errors_theta, ANG_VELS_THR_L)
        ang_vels_errors[theta]['precision']['loose'] = p
        ang_vels_errors[theta]['recall']['loose'] = r
        ang_vels_errors[theta]['f1']['loose'] = f1
        ang_vels_precisions['loose'].append(p)
        ang_vels_recalls['loose'].append(r)
        ang_vels_f1s['loose'].append(f1)

        # ANGULAR ACCELERATION ERRORS
        # --------------------------------------------------- #

        gt_theta_ang_accs = ground_truth['ang_accs'][theta]

        ang_accs_errors[theta] = {}

        ang_accs_errors_theta = np.abs(np.asarray(gt_theta_ang_accs) - np.asarray(ang_accs[theta]))
        ang_accs_errors[theta]['framewise_error'] = np.ndarray.tolist(ang_accs_errors_theta)
        ang_accs_errors[theta]['error_mean'] = np.nanmean(ang_accs_errors_theta)
        ang_accs_errors[theta]['error_median'] = np.nanmedian(ang_accs_errors_theta)

        ang_accs_error_means.append(ang_accs_errors[theta]['error_mean'])
        ang_accs_error_medians.append(ang_accs_errors[theta]['error_median'])

        ang_accs_icc = icc2_1(gt_theta_ang_accs, ang_accs[theta])
        ang_accs_errors[theta]['ICC'] = ang_accs_icc

        ang_accs_errors[theta]['precision'] = {}
        ang_accs_errors[theta]['recall'] = {}
        ang_accs_errors[theta]['f1'] = {}

        p, r, f1 = p_r_f1_score(ang_accs_errors_theta, ANG_ACCS_THR_T)
        ang_accs_errors[theta]['precision']['tight'] = p
        ang_accs_errors[theta]['recall']['tight'] = r
        ang_accs_errors[theta]['f1']['tight'] = f1
        ang_accs_precisions['tight'].append(p)
        ang_accs_recalls['tight'].append(r)
        ang_accs_f1s['tight'].append(f1)

        p, r, f1 = p_r_f1_score(ang_accs_errors_theta, ANG_ACCS_THR_L)
        ang_accs_errors[theta]['precision']['loose'] = p
        ang_accs_errors[theta]['recall']['loose'] = r
        ang_accs_errors[theta]['f1']['loose'] = f1
        ang_accs_precisions['loose'].append(p)
        ang_accs_recalls['loose'].append(r)
        ang_accs_f1s['loose'].append(f1)

    angle_errors['error_mean_mean'] = np.sum(angle_error_means) / len(angle_error_means)
    angle_errors['error_mean_median'] = np.sum(angle_error_medians) / len(angle_error_medians)

    ang_vels_errors['error_mean_mean'] = np.sum(ang_vels_error_means) / len(ang_vels_error_means)
    ang_vels_errors['error_mean_median'] = np.sum(ang_vels_error_medians) / len(ang_vels_error_medians)

    ang_accs_errors['error_mean_mean'] = np.sum(ang_accs_error_means) / len(ang_accs_error_means)
    ang_accs_errors['error_mean_median'] = np.sum(ang_accs_error_medians) / len(ang_accs_error_medians)

    angle_errors['average_precision'] = {
        'tight': np.sum(angle_precisions['tight']) / len(angle_precisions['tight']),
        'loose': np.sum(angle_precisions['loose']) / len(angle_precisions['loose'])
    }
    angle_errors['average_recall'] = {
        'tight': np.sum(angle_recalls['tight']) / len(angle_recalls['tight']),
        'loose': np.sum(angle_recalls['loose']) / len(angle_recalls['loose'])
    }
    angle_errors['average_f1'] = {
        'tight': np.sum(angle_f1s['tight']) / len(angle_f1s['tight']),
        'loose': np.sum(angle_f1s['loose']) / len(angle_f1s['loose'])
    }

    ang_vels_errors['average_precision'] = {
        'tight': np.sum(ang_vels_precisions['tight']) / len(ang_vels_precisions['tight']),
        'loose': np.sum(ang_vels_precisions['loose']) / len(ang_vels_precisions['loose'])
    }
    ang_vels_errors['average_recall'] = {
        'tight': np.sum(ang_vels_recalls['tight']) / len(ang_vels_recalls['tight']),
        'loose': np.sum(ang_vels_recalls['loose']) / len(ang_vels_recalls['loose'])
    }
    ang_vels_errors['average_f1'] = {
        'tight': np.sum(ang_vels_f1s['tight']) / len(ang_vels_f1s['tight']),
        'loose': np.sum(ang_vels_f1s['loose']) / len(ang_vels_f1s['loose'])
    }

    ang_accs_errors['average_precision'] = {
        'tight': np.sum(ang_accs_precisions['tight']) / len(ang_accs_precisions['tight']),
        'loose': np.sum(ang_accs_precisions['loose']) / len(ang_accs_precisions['loose'])
    }
    ang_accs_errors['average_recall'] = {
        'tight': np.sum(ang_accs_recalls['tight']) / len(ang_accs_recalls['tight']),
        'loose': np.sum(ang_accs_recalls['loose']) / len(ang_accs_recalls['loose'])
    }
    ang_accs_errors['average_f1'] = {
        'tight': np.sum(ang_accs_f1s['tight']) / len(ang_accs_f1s['tight']),
        'loose': np.sum(ang_accs_f1s['loose']) / len(ang_accs_f1s['loose'])
    }

    metric_dict = {
        'angle_metrics': angle_errors,
        'ang_vels_metrics': ang_vels_errors,
        'ang_accs_metrics': ang_accs_errors,
    }

    return metric_dict