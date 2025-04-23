import matplotlib.pyplot as plt
import numpy as np
import cv2  # type: ignore
import json
import os
from utils.data_utils import make_path
from metrics.metrics import frechet_distance, z_score_normalise, dtw_distance
from metrics.angles import remove_jump_discontinuity, butterworth_derivative

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    # Retrieve FPS from the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV format) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames, fps

def overlay_keypoints_on_video(video_path, output_dir, model_type, subject, camera, keypoints, pred_conf, connections, plot_confidence, confidence):
    figscale = 2.5

    video_frames, fps = load_video(video_path)
    filename = os.path.splitext(os.path.basename(video_path))[0]

    fig = plt.figure(figsize=(5*figscale, 5*figscale))

    # Remove boundary
    plt.axis('off')

    output_frames = []

    # Calculate width and height of video
    height, width, _ = video_frames[0].shape

    # Loop through each frame of keypoints
    for frame_index, frame_keypoints in enumerate(keypoints):
        # clear plot
        plt.clf()

        # Remove axes and adjust figure position
        ax = plt.gca()
        ax.set_axis_off()  # Hide axis
        ax.set_position([0, 0, 1, 1])  # Remove padding

        # Plot each joint
        for joint_i in range(0, len(frame_keypoints)):
            if pred_conf[frame_index][joint_i] > confidence:
                x = int(frame_keypoints[joint_i][1] * width)
                y = int(frame_keypoints[joint_i][0] * height)
                if plot_confidence:
                    # Colour based on confidence
                    colour_conf = pred_conf[frame_index][joint_i]
                    # Bounding outliers between 0 and 1
                    if colour_conf > 1: colour_conf = 1
                    elif colour_conf < 0: colour_conf = 0
                    
                    plt.plot(x, y, '.', markersize=pred_conf[frame_index][joint_i]*7*figscale, color=(1-colour_conf,colour_conf,0))
                else:
                    plt.plot(x, y, '.', markersize=6*figscale, color=(0, 1, 0))

        # Plot each connection
        for connection in connections:
            if pred_conf[frame_index][connection[0]] > confidence and pred_conf[frame_index][connection[1]] > confidence:
                x1 = int(frame_keypoints[connection[0]][1] * width)
                y1 = int(frame_keypoints[connection[0]][0] * height)
                x2 = int(frame_keypoints[connection[1]][1] * width)
                y2 = int(frame_keypoints[connection[1]][0] * height)
                if plot_confidence:
                    # Colour based on confidence
                    connection_confidence = 0.5 * (pred_conf[frame_index][connection[0]] + pred_conf[frame_index][connection[1]]) # Mean joint confidence of the two connection joints
                    # Bounding outliers between 0 and 1
                    if connection_confidence > 1: connection_confidence = 1
                    elif connection_confidence < 0: connection_confidence = 0

                    plt.plot([x1, x2], [y1, y2], '-', linewidth=0.5*figscale, color=(1-connection_confidence, connection_confidence,0))
                else:
                    plt.plot([x1, x2], [y1, y2], '-', linewidth=0.5*figscale, color=(0, 0.8, 0))

        # Display the image with predicted joints
        image = video_frames[frame_index].astype(np.uint8)
        plt.imshow(image)

        fig.canvas.draw()  # Render the figure
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Convert RGB to BGR for OpenCV compatibility
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if frame_index == 0:
            # Create a video writer
            w_width, w_height, _ = img_bgr.shape
            output_path = make_path(output_dir, filename, subject, camera, model_type, "mp4")
            # Ensure the directory exists before writing the video
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_width, w_height))

        # Write frame
        video_writer.write(img_bgr)

    video_writer.release()
    print(f"Video saved as {output_path}")

# VISUALISATION: PRODUCE GRAPHS COMPARING PREDICTION TO GROUND TRUTH
def plot_comparison():

    # ------------------------------------- #
    # Change as desired
    filename = "warmup_7"
    subject = "s05"
    camera = "65906101"
    model = "yolopose"
    theta = "8"
    frames = [140, 240]
    # ------------------------------------- #

    ground_truth_path = make_path(
        save_dir = "data/fit3d_train",
        file_path = filename,
        subject = subject,
        camera = camera,
        model_type = "ground_truth",
        extension = "json"
    )
    
    predictions_path = make_path(
        save_dir = "output/fit3d_train/predictions",
        file_path = filename,
        subject = subject,
        camera = camera,
        model_type = model,
        extension = "json"
    )

    metrics_path = make_path(
        save_dir = "output/fit3d_train/metrics",
        file_path = filename,
        subject = subject,
        camera = camera,
        model_type = model,
        extension = "json"
    )

    with open(ground_truth_path) as gt_file:
        ground_truth = json.load(gt_file)

    with open(predictions_path) as p_file:
        predictions = json.load(p_file)

    with open(metrics_path) as m_file:
        metrics = json.load(m_file)

    gt_angle = ground_truth["angles"][theta]
    predicted_angle = predictions["angles"][theta]
    frame_difference = len(predicted_angle) - len(gt_angle)
    print(f"{frame_difference=}")
    predicted_angle = predicted_angle[:-frame_difference]
    metrics_angles = metrics["angle_metrics"][theta]["framewise_error"]

    gt_ang_vels = ground_truth["ang_vels"][theta]
    predicted_ang_vels = predictions["ang_vels"][theta][:-frame_difference]
    metrics_ang_vels = metrics["ang_vels_metrics"][theta]["framewise_error"]

    gt_ang_accs = ground_truth["ang_accs"][theta]
    predicted_ang_accs = predictions["ang_accs"][theta][:-frame_difference]
    metrics_ang_accs = metrics["ang_accs_metrics"][theta]["framewise_error"]

    x_values = range(len(gt_angle))

    # Create the plot
    plt.figure(figsize=(6, 3.75))
    plt.plot(x_values[frames[0]:frames[1]], gt_angle[frames[0]:frames[1]], linestyle='-', label="GT Angle", c="sandybrown")
    plt.plot(x_values[frames[0]:frames[1]], predicted_angle[frames[0]:frames[1]], linestyle='--', label="Pred. Angle", c="peru")
    #plt.plot(x_values[frames[0]:frames[1]], metrics_angles[frames[0]:frames[1]], linestyle=':', label="Error", c="saddlebrown")

    # Labels and title
    plt.xlabel("Frame")
    plt.ylabel("Angle (rad)")
    plt.title(f"{model} - Angles")

    # Show legend
    #plt.legend()

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.show()

    plt.figure(figsize=(6, 3.75))
    plt.plot(x_values[frames[0]:frames[1]], gt_ang_vels[frames[0]:frames[1]], linestyle='-', label="GT Ang. Vel.", c="mediumorchid")
    plt.plot(x_values[frames[0]:frames[1]], predicted_ang_vels[frames[0]:frames[1]], linestyle='--', label="Pred. Ang. Vel.", c="darkorchid")
    #plt.plot(x_values[frames[0]:frames[1]], metrics_ang_vels[frames[0]:frames[1]], linestyle=':', label="Error", c="blueviolet")

    # Labels and title
    plt.xlabel("Frame")
    plt.ylabel("Anglular Velocity (rad/s)")
    plt.title(f"{model} - Ang. Vels.")

    # Show legend
    #plt.legend()

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.show()

    plt.figure(figsize=(6, 3.75))
    plt.plot(x_values[frames[0]:frames[1]], gt_ang_accs[frames[0]:frames[1]], linestyle='-', label="GT. Ang. Acc.", c="olivedrab")
    plt.plot(x_values[frames[0]:frames[1]], predicted_ang_accs[frames[0]:frames[1]], linestyle='--', label="Pred. Ang. Acc.", c="forestgreen")
    #plt.plot(x_values[frames[0]:frames[1]], metrics_ang_accs[frames[0]:frames[1]], linestyle=':', label="Error", c="darkgreen")

    # Labels and title
    plt.xlabel("Frame")
    plt.ylabel("Angular Acceleration (rad/s/s)")
    plt.title(f"{model} - Ang. Accs.")

    # Show legend
    #plt.legend()

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.show()

    plt.figure(figsize=(6, 3.75))
    plt.plot(x_values[frames[0]:frames[1]], metrics_angles[frames[0]:frames[1]], linestyle=':', label="Angle Error (rad)", c="saddlebrown")
    plt.plot(x_values[frames[0]:frames[1]], metrics_ang_vels[frames[0]:frames[1]], linestyle=':', label="Ang. Vel. Error (rad/s)", c="blueviolet")
    plt.plot(x_values[frames[0]:frames[1]], metrics_ang_accs[frames[0]:frames[1]], linestyle=':', label="Ang. Acc. Error (rad/s/s)", c="darkgreen")

    # Labels and title
    plt.xlabel("Frame")
    plt.ylabel("Error")
    plt.title(f"{model} - Errors")

    # Show legend
    #plt.legend()

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.show()