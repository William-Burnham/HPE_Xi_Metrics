# 📄 HPE Xi Metrics
**Code for**: _"Benchmark Angular Metrics for 2D Human Pose Estimation"_  
**Authors**: William Burnham, Nicholas Costen, Moi Hoon Yap, Rick Mills, Johnny Parr, Emma Vardy, Sean Maudsley-Barton  
**Published in**: As yet unpublished
[📄 Read Paper]()

---

## 🧠 Abstract

This repository contains the code used in our paper, "_Benchmark Angular Metrics for 2D Human Pose Estimation_".

Human Pose Estimation (HPE) offers a promising alternative to traditional motion capture systems for clinical applications due to its affordability and ease of use. However, existing evaluation metrics for 2D HPE focus primarily on key- point positional accuracy, neglecting the angular information that is critical for biomechanical and clinical analysis. In this paper, we introduce a set of benchmark angular metrics — referred to as ξ metrics — that assess a model’s ability to predict joint angles and their derivatives (angular velocity and acceleration) from 2D pose data. Using a synchronised RGB and Vicon motion capture dataset, we evaluate the performance of four leading HPE models and analyse their suitability for clinical application. Our findings show that MoveNet is comparable with electrogoniometers in angles and angular velocity, with Average Precision in angles (AP<sup>loose</sup><sub>θ</sub>) of 0.701 and Average Precision of angular velocity (AP<sup>loose</sup><sub>ω</sub>) of 0.781, and has the lowest Mean Absolute Error in angles and angular derivatives. However, none of the models perform comparably with the accuracy of marker-based motion capture systems. This research highlights the limitations of current models in predicting angular dynamics and underscores the need for a clinical focus in HPE research to develop new models that optimise for angular metrics.

---

## 📁 Repository Structure

```
├── data/                   # Download dataset
├── metrics/                # Metrics code
├── models/                 # Models code
│   └── hpe_models/
│       ├── hpe_model.py    # Base class
│       ├── mediapipe.py    # MediaPipe Pose
│       ├── mmpose_model.py # MMPose functionality (not used)
│       ├── movenet.py      # MoveNet models
│       └── yolopose.py     # YOLO-Pose
├── output/                 # Generated outputs (predictions, plots, metrics)
├── requirements/           # Organised requirements
├── utils/                  # Utilities
├── configuration.json      # Configuration file
├── hpe_xi_metrics.py       # Main script
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── LICENSE                 # License info
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/William-Burnham/HPE_Xi_Metrics.git
cd HPE_Xi_Metrics
```

### 2. Set Up Environment

It's recommended to use a virtual environment:

```bash
conda create --name xi_metrics python=3.9
conda activate xi_metrics
```

Install requirements with pip:

```bash
pip install -r requirements.txt
```

If running on GPU:
```bash
pip install torch torchvision
```

If running on CPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Download Dataset

We use the Fit3D dataset for our experiment (available [here](https://fit3d.imar.ro/)). Any other dataset must be configured in accordance with the format of Fit3D:

```
└── dataset/                            # Dataset name
   └── subset/                          # Subset name (eg. 'train')
        ├── subject_0/                  # Subject ID
        │   ├── camera_parameters/
        │   │   ├ camera_0/
        │   │   │   ├ action_0.json     # .json file containing intrinsic and extrinsic camera parameters of the corresponding video
        │   │   │   ├ action_1.json
        │   │   │   │   ...
        │   │   │   └ action_n.json
        │   │   ├ camera_1/
        │   │   │   ...
        │   │   └ camera_n/
        │   ├── joints3d_25/
        │   │   ├ action_0.json         # .json file containing 3D joint position data of the corresponding video
        │   │   ├ action_1.json
        │   │   │   ...
        │   │   └ action_n.json        
        │   └── videos/
        │       ├ camera_0/
        │       │   ├ action_0.mp4      # .mp4 file containing action video sequence
        │       │   ├ action_1.mp4
        │       │   │   ...
        │       │   └ action_n.mp4
        │       ├ camera_1/
        │       │   ...
        │       └ camera_n/
        ├── subject_1/
        │   ...
        └── subject_n/
```

### 4. Tune Configuration

This code has multiple layers of functionality that can be tuned via the `configuration.json` file.

```python
{
    "run_on_cpu"                # (Boolean) Run code on CPU

    "data_root"                 # Root folder of data
    "dataset_name"              # Dataset folder name (within data_root)
    "subset"                    # Data subset within (data_root/dataset_name)

    "frame_w"                   # (Integer) Pixel width of video files
    "frame_h"                   # (Integer) Pixel height of video files
    "fps"                       # (Integer) Frame rate of video files

    "do_compute_gt"             # (Boolean) Compute the angles and angular derivatives from the ground truth dataset ('true' on first time using a new dataset)
    "gt_dir"                    # Directory for ground truth calculations to be saved to / loaded from

    "do_compute_predictions"    # (Boolean) Compute model predictions on dataset with selected models (keypoints; confidences; angles; angular derivatives)
    "predictions_output_folder" # Directory for predictions to be saved to / loaded from
    "do_save_overlay"           # (Boolean) NOT RECOMMENDED - VERY SLOW: Save videos with overlaid skeletons for each prediction
    "do_plot_confidence"        # (Boolean) Use joint colour to visualise prediction confidence of each skeleton overlay
    "overlay_output_folder"     # Directory for overlaid videos to be saved to

    "do_compute_metrics"        # (Boolean) Compute metrics from the predictions of selected models (ensure predictions are computed first)
    "do_ankles"                 # (Boolean) NOT RECOMMENDED - MEDIAPIPE EXCLUSIVE: Include ankles in metric calculations
    "do_transverse_angles"      # (Boolean) Include internal hip and external shoulder in metric calculations
    "metrics_output_folder"     # Directory for metrics to be saved to

    "mediapipe_pose"            # (Boolean) Select MediaPipe Pose model
    "movenet_lightning"         # (Boolean) Select MoveNet Lightning model
    "movenet_thunder"           # (Boolean) Select MoveNet Thunder model
    "yolopose"                  # (Boolean) Select YOLO-Pose model
}
```

On each first run of a new dataset, use `"do_compute_gt": true`. This will compute and save all of the ground truth 2D projected keypoints, angles and angular derivatives. In further runs, there is no need to use `"do_compute_gt"` unless the dataset changes, as the output will be saved.

Use `"do_compute_predictions": true` to compute the keypoints, angles and angular derivatives with the selected models. Again, if running multiple tests on the same model predictions, only compute these once unless the model or dataset changes, as the output will be saved.

Use the bottom four choices to select which models to run.

### 5. Run the Code

Example:

```bash
python hpe_xi_metrics
```

---

## 🔧 Add New Models

### 1. Setup

All new models should inherit `HPE_Model()` and be stored as a Python file in `models/hpe_models/`.

### 2. Attributes

New models should define the following attributes in their constructors:

```python
self.model_type         # Name of model
self.KEYPOINT_DICT      # A dictionary of keypoint names and indices
self.CONNECTIONS        # A list of joint connections (bones) used in creating overlays
```

Example:

```python
self.model_type = "new_model"

# INDICES AND KEYPOINT NAMES ARE IMPORTANT
# Indices must align with the corresponding joint index within 'keypoints' and 'confidences' (see 3. Predict Function).
# Below are the names of all 14 keypoint names that will align with Fit3D's Vicon data:
self.KEYPOINT_DICT = {
    'left_foot_index': 0,
    'right_foot_index': 1,
    'left_ankle': 2,
    'right_ankle': 3,
    'left_knee': 4,
    'right_knee': 5,
    'left_hip': 6,
    'right_hip': 7,
    'left_shoulder': 8,
    'right_shoulder': 9,
    'left_elbow': 10,
    'right_elbow': 11,
    'left_wrist': 12,
    'right_wrist': 13
}

# e.g [11,13] connects "right_elbow" and "right_wrist"
self.CONNECTIONS = [[0,1],[2,3],[2,4],[3,5],[4,6],[5,7],[6,7],[6,8],[7,9],[8,9],[8,10],[9,11],[10,12],[11,13]]
```

### 3. Predict Function

New models should define a `predict()` function. This function runs the model over a whole video, returning its keypoint predictions and confidences.

The function requires one argument, `file_path`, which is a string storing the file path of the chosen video.

The function should return a tuple: `keypoints, confidences`.

`keypoints` is a 3D list of predicted joint positions `[y,x]` nested for each joint within each frame of the video (helper functionality is provided in `utils.kp_utils.reposition_keypoints()` to swap the positions of x and y, or flip their values across the frame if needed).

`confidences` is a 2D list of the joint prediction confidence `c` nested for each joint within each frame of the video.

The joint indices in `keypoints` and `confidences` must correspond with the joint indices defined in `self.KEYPOINT_DICT`.

### 4. Add Model to Main Code

In `configuration.json` add a new choice for your model:

```json
"your_model": true
```

In `hpe_xi_metrics.main()` add a new section for the new model following the template:

```python
    # Your model name
    if config["your_model"]:
        # Load model
        from models.hpe_models.your_model_name import YourModelClass
        model = YourModelClass()
        if config["do_compute_predictions"]:
            # Run full inference
            run_model_on_dataset(model, config)
        if config["do_compute_metrics"]:
            # Get model metrics
            compute_metrics(model, config)
```

Now, when running the code with `your_model` selected in the configuration file, it will be used in all relevant computations.

---

## 📊 Results

### 📈 Average Precision (AP) Comparison on Fit3D Dataset

**AP<sup>tight</sup>** = tight threshold, **AP<sup>loose</sup>** = loose threshold

| Model             | APθ tight | APω tight | APα tight | APθ loose | APω loose | APα loose |
|-------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| MediaPipe Pose    | 0.387     | 0.66      | **0.445** | 0.666     | 0.758     | **0.581** |
| MoveNet Lightning | 0.377     | 0.567     | 0.227     | 0.663     | 0.707     | 0.386     |
| MoveNet Thunder   | 0.401     | **0.661** | 0.307     | **0.701** | **0.781** | 0.494     |
| YOLO-Pose         | **0.417** | 0.589     | 0.294     | 0.691     | 0.714     | 0.453     |


### 📈 Average Recall (AR) Comparison on Fit3D Dataset

**AR<sup>tight</sup>** = tight threshold, **AR<sup>loose</sup>** = loose threshold

| Model             | ARθ tight | ARω tight | ARα tight | ARθ loose | ARω loose | ARα loose |
|-------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| MediaPipe Pose    | 0.901     | 0.997     | 0.993     | 0.973     | 0.997     | 0.995     |
| MoveNet Lightning | 0.944     | **1.000** | **1.000** | 0.990     | **1.000** | **1.000** |
| MoveNet Thunder   | 0.923     | **1.000** | **1.000** | 0.987     | **1.000** | **1.000** |
| YOLO-Pose         | **0.950** | 0.999+    | 0.999+    | **0.991** | 0.999+    | 0.999+    |


### 📈 Average F₁ Score Comparison on Fit3D Dataset

**AF₁<sup>tight</sup>** = tight threshold, **AF₁<sup>loose</sup>** = loose threshold

| Model             | AF₁θ tight | AF₁ω tight | AF₁α tight | AF₁θ loose | AF₁ω loose | AF₁α loose |
|-------------------|------------|------------|------------|------------|------------|------------|
| MediaPipe Pose    | 0.487      | 0.770      | **0.576**  | 0.752      | 0.844      | **0.701**  |
| MoveNet Lightning | 0.482      | 0.700      | 0.533      | 0.751      | 0.809      | 0.533      |
| MoveNet Thunder   | 0.499      | **0.775**  | 0.452      | 0.780      | **0.863**  | 0.637      |
| YOLO-Pose         | **0.534**  | 0.720      | 0.438      | **0.781**  | 0.816      | 0.601      |


### 📉 Mean Absolute Error (MAE) Comparison on Fit3D Dataset

| Model             | MAEθ      | MAEω      | MAEα     |
|-------------------|-----------|-----------|----------|
| MediaPipe Pose    | 0.201     | 0.862     | 14.5     |
| MoveNet Lightning | 0.203     | 1.070     | 19.4     |
| MoveNet Thunder   | **0.178** | **0.768** | **13.5** |
| YOLO-Pose         | 0.208     | 0.953     | 16.5     |

---

## 📚 Citation

If you use this code, please cite:

```bibtex
As yet unpublished
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋 Acknowledgments

This research is funded by Manchester Metropolitan University PhD Studentship. Part of the research is sponsored by EPSRC funding (EP/Z00084X/1).