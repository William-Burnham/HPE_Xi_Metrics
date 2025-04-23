import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from alive_progress import alive_bar # type: ignore

from utils.data_utils import load_file_paths, save_predictions, compute_ground_truth, find_json_files, save_metrics
from utils.plot_utils import overlay_keypoints_on_video

from metrics.angles import calculate_angles
from metrics.metrics import compare_to_gt

import traceback

def main(config):

    # Compute 2D joints and ground truth data for metrics
    # --------------------------------------------------- #
    if config['do_compute_gt']:
        # Compute ground truth from dataset and save to json
        # (only required on first run, unless there are changes to the dataset)
        compute_ground_truth(config)

    # Run selected models on all files within dataset:
    # --------------------------------------------------- #
    
    # HRNet
    #if config['hrnet']:
    #    # Load hrnet model
    #    from models.hpe_models.mmpose_model import MMPose_Model
    #    model = MMPose_Model(model_type='hrnet')
    #    if config["do_compute_predictions"]:
    #        # Run full inference
    #        run_model_on_dataset(model, config)
    #    if config["do_compute_metrics"]:
    #        # Get model metrics
    #        compute_metrics(model, config)

    # MediaPipe Pose
    if config["mediapipe_pose"]:
        # Load mediapipe model
        from models.hpe_models.mediapipe import MediaPipe_Model
        model = MediaPipe_Model()
        if config["do_compute_predictions"]:
            # Run full inference
            run_model_on_dataset(model, config)
        if config["do_compute_metrics"]:
            # Get model metrics
            compute_metrics(model, config)

    # MoveNet Lightning
    if config["movenet_lightning"]:
        # Load movenet model
        from models.hpe_models.movenet import MoveNet_Model
        model = MoveNet_Model(model_type="movenet_lightning")
        if config["do_compute_predictions"]:
            # Run full inference
            run_model_on_dataset(model, config)
        if config["do_compute_metrics"]:
            # Get model metrics
            compute_metrics(model, config)

    # MoveNet Thunder
    if config["movenet_thunder"]:
        # Load movenet model
        from models.hpe_models.movenet import MoveNet_Model
        model = MoveNet_Model(model_type="movenet_thunder")
        if config["do_compute_predictions"]:
            # Run full inference
            run_model_on_dataset(model, config)
        if config["do_compute_metrics"]:
            # Get model metrics
            compute_metrics(model, config)

    # YoloPose (Ultralytics)
    if config["yolopose"]:
        # Load yolopose model
        from models.hpe_models.yolopose import Yolo_Model
        model = Yolo_Model()
        if config["do_compute_predictions"]:
            # Run full inference
            run_model_on_dataset(model, config)
        if config["do_compute_metrics"]:
            # Get model metrics
            compute_metrics(model, config)

def run_model_on_dataset(model, config):
    # Nested folder query
    # Layer 1: per subject
    subjects_dir = os.path.join(config['data_root'], config['dataset_name'], config['subset'])
    subject_ids = os.listdir(subjects_dir)
    for si, subject in enumerate(subject_ids):
        # Layer 2: per camera
        cameras_dir = os.path.join(subjects_dir, subject, 'videos')
        camera_ids = os.listdir(cameras_dir)
        for ci, camera in enumerate(camera_ids):
            # Layer 3: per video / per exercise
            videos_dir = os.path.join(cameras_dir, camera)
            file_paths = load_file_paths(videos_dir)

            for fi, file_path in enumerate(file_paths):
                
                print(f"{model.get_model_type()} prediction: {(si*len(file_paths)*len(camera_ids))+(ci*len(file_paths))+fi+1}/{len(file_paths)*len(camera_ids)*len(subject_ids)}")
                # Run model on video
                start = time.time()
                keypoints, confidences = model.predict(file_path=file_path)
                duration = time.time() - start
                # Calculate angles and angular derivatives from keypoint data
                angles, ang_vels, ang_accs = calculate_angles(keypoints, model.get_kpd(), config['fps'])

                save_output(model, keypoints, angles, ang_vels, ang_accs, confidences, file_path, subject, camera, duration, config)

def compute_metrics(model, config):
    # Get the list of all prediction file names
    directory_path = os.path.join(config["predictions_output_folder"], model.get_model_type())
    prediction_paths = find_json_files(directory_path)

    # Initialise dictionaries
    mu_mae = {
        "angle_metrics": [],
        "ang_vels_metrics": [],
        "ang_accs_metrics": []
    }
    xi_metrics = {}
    xi_metrics["angle_metrics"] = {
        "average_precision": {"tight": [], "loose": []},
        "average_recall": {"tight": [], "loose": []},
        "average_f1": {"tight": [], "loose": []}
    }
    xi_metrics["ang_vels_metrics"] = {
        "average_precision": {"tight": [], "loose": []},
        "average_recall": {"tight": [], "loose": []},
        "average_f1": {"tight": [], "loose": []}
    }
    xi_metrics["ang_accs_metrics"] = {
        "average_precision": {"tight": [], "loose": []},
        "average_recall": {"tight": [], "loose": []},
        "average_f1": {"tight": [], "loose": []}
    }

    # Progress bar
    with alive_bar(len(prediction_paths)+1, title="Computing metrics...", length=16) as bar:
        
        # Iterate all predictions
        for path in prediction_paths:
            # Load file contents
            with open(path) as file:
                pred_data = json.load(file)        
            keypoints = pred_data['keypoints']

            path_parts = path.split(os.sep)
            subject = path_parts[-3]
            camera = path_parts[-2]
            action = os.path.splitext(path_parts[-1])[0]

            # ----------------------------------------------------------------- #
            # Exclude these actions (change if desired) :
            if action in ["burpees", "diamond_pushup", "man_maker", "mule_kick", "pushup", "warmup_1"]:
                bar()
                continue
            # ----------------------------------------------------------------- #

            # Compute metrics for this prediction
            metric_dict = compare_to_gt(
                model=model,
                keypoints=pred_data['keypoints'],
                angles=pred_data['angles'],
                ang_vels=pred_data['ang_vels'],
                ang_accs=pred_data['ang_accs'],
                file_path=path,
                subject=subject,
                camera=camera,
                config=config
            )

            # Save to file
            save_metrics(
                model_type=model.get_model_type(),
                metric_dict=metric_dict,
                save_dir=config['metrics_output_folder'],
                file_path=path,
                subject=subject,
                camera=camera
            )

            # Append average MAE across joints
            mu_mae["angle_metrics"].append(metric_dict["angle_metrics"]["error_mean_mean"])
            mu_mae["ang_vels_metrics"].append(metric_dict["ang_vels_metrics"]["error_mean_mean"])
            mu_mae["ang_accs_metrics"].append(metric_dict["ang_accs_metrics"]["error_mean_mean"])

            # Append average Xi metrics across joints
            for xi_key, xi_value in xi_metrics.items():
                for metric_key, metric_value in xi_value.items():
                    for threshold_key in metric_value.keys():
                        xi_metrics[xi_key][metric_key][threshold_key].append(metric_dict[xi_key][metric_key][threshold_key])

            bar()

        # Save the full metrics to a file - before averaging
        full_metric_dict = {
            "mu_mae": mu_mae,
            "xi_metrics": xi_metrics
        }
        save_path = os.path.join(config["metrics_output_folder"], model.get_model_type(), "full_metrics.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, mode='w') as file:
            json.dump(full_metric_dict, file, indent=4)

        # Calculate averages
        mu_mae["angle_metrics"] = np.mean(mu_mae["angle_metrics"])
        mu_mae["ang_vels_metrics"] = np.mean(mu_mae["ang_vels_metrics"])
        mu_mae["ang_accs_metrics"] = np.mean(mu_mae["ang_accs_metrics"])

        for xi_key, xi_value in xi_metrics.items():
            for metric_key, metric_value in xi_value.items():
                for threshold_key, value in metric_value.items():
                    xi_metrics[xi_key][metric_key][threshold_key] = np.mean(value)

        bar()

    # Save the averaged metrics
    averaged_metric_dict = {
        "mu_mae": mu_mae,
        "xi_metrics": xi_metrics
    }
    save_path = os.path.join(config["metrics_output_folder"], model.get_model_type(), "avg_metrics.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode='w') as file:
        json.dump(averaged_metric_dict, file, indent=4)

def save_output(model, keypoints, angles, ang_vels, ang_accs, confidences, file_path, subject, camera, duration, config):
    save_predictions(
        model_type=model.get_model_type(),
        keypoint_dict=model.get_kpd(),
        keypoints=keypoints,
        angles=angles,
        ang_vels=ang_vels,
        ang_accs=ang_accs,
        confidences=confidences,
        save_dir=config["predictions_output_folder"],
        file_path = file_path,
        subject = subject,
        camera = camera,
        duration=duration
    )
        
    if config["do_save_overlay"]: overlay_keypoints_on_video(
        video_path=file_path,
        output_dir=config["overlay_output_folder"],
        model_type=model.get_model_type(),
        subject = subject,
        camera = camera,
        keypoints=keypoints,
        pred_conf=confidences,
        connections=model.get_connections(),
        plot_confidence=config["do_plot_confidence"],
        confidence=model.get_confidence()
    )

if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser(description='HPE_Comparative_Study')

        parser.add_argument('--config_json', '-config', default='configuration.json', type=str)

        args = parser.parse_args()

        config_file = args.config_json
        with open(config_file) as json_file:
            config = json.load(json_file)

        if config['run_on_cpu']:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        main(config)

    except Exception as e:
        
        print("Caught an exception:")
        traceback.print_exc()