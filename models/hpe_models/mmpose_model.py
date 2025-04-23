import os
import cv2
import torch
import numpy as np
import time
from alive_progress import alive_bar # type: ignore

import mmcv # type: ignore
from mmcv import imread # type: ignore
from mmengine.registry import init_default_scope # type: ignore

from mmpose.apis import inference_topdown # type: ignore
from mmpose.apis import init_model as init_pose_estimator # type: ignore
from mmpose.evaluation.functional import nms # type: ignore
from mmpose.structures import merge_data_samples # type: ignore

try:
    import mmdet # type: ignore
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from mmdet.apis.inference import inference_detector, init_detector # type: ignore

from models.hpe_models.hpe_model import HPE_Model
from utils.data_utils import download_file
from utils.kp_utils import reposition_keypoints

class MMPose_Model(HPE_Model):
    def __init__(self, model_type="HRNet", detector="Faster-RCNN"):
        HPE_Model.__init__(self)

        self.model_type = model_type
        self.detector = detector

        # Select paths for config and checkpoint
        if self.model_type.lower() == "hrnet":
            self.POSE_CONFIG_URL = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
            self.POSE_CONFIG_PATH = "configs/body_2d_keypoint/topdown_heatmap/coco/hrnet_w32_coco_256x192.py"
            self.POSE_CHECKPOINT = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"

        else:
            raise Exception(f"(MMPose_Model): Unsupported model type {model_type}")
        
        # Select detector model
        if self.detector.lower() == "rtmdet":

            self.DET_CONFIG_URL = "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py"
            self.DET_CONFIG_PATH = "configs/rtmdet/rtmdet_l_8xb32-300e_coco.py"
            self.DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
            
            download_file(
                "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/default_runtime.py",
                "configs/_base_/default_runtime.py"
            )
            download_file(
                "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/rtmdet/rtmdet_tta.py",
                "configs/rtmdet/rtmdet_tta.py"
            )
            download_file(
                "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/datasets/coco_detection.py",
                "configs/_base_/datasets/coco_detection.py"
            )
            download_file(
                "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/schedules/schedule_1x.py",
                "configs/_base_/schedules/schedule_1x.py"
            )

        elif self.detector.lower() == "faster-rcnn":

            self.DET_CONFIG_URL = "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/faster_rcnn/faster_rcnn_r50_fpn_coco.py"
            self.DET_CONFIG_PATH = "configs/faster_rcnn/faster_rcnn_r50_fpn_coco.py"
            self.DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
            
            download_file(
                "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/default_runtime.py",
                "configs/_base_/default_runtime.py"
            )
            download_file(
                "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/models/faster-rcnn_r50_fpn.py",
                "configs/_base_/models/faster-rcnn_r50_fpn.py"
            )
            download_file(
                "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/datasets/coco_detection.py",
                "configs/_base_/datasets/coco_detection.py"
            )
            download_file(
                "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/_base_/schedules/schedule_1x.py",
                "configs/_base_/schedules/schedule_1x.py"
            )

        else:
            raise Exception(f"(MMPose_Model): Unsupported detector {detector}")


        # Ensure config files are downloaded
        download_file(self.POSE_CONFIG_URL, self.POSE_CONFIG_PATH)
        download_file(self.DET_CONFIG_URL, self.DET_CONFIG_PATH)

        # Load the selected model in two parts, detector and pose estimator
        self.detector, self.pose_estimator = self.load_model(self.POSE_CONFIG_PATH, self.POSE_CHECKPOINT, self.DET_CONFIG_PATH, self.DET_CHECKPOINT)
        
        self.KEYPOINT_DICT = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }

        self.CONNECTIONS = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[5,11],[6,8],[6,12],[7,9],[8,10],[11,12],[11,13],[12,14],[13,15],[14,16]]

    def load_model(self, pose_config, pose_checkpoint, det_config, det_checkpoint):
        # Build detector and pose estimator
        # MMPose tutorial: https://colab.research.google.com/drive/1rrCq6uPq6MhbNoslvBLqhljVKhqUO3RB#scrollTo=JjTt4LZAx_lK
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
        
        # build detector
        detector = init_detector(
            det_config,
            det_checkpoint,
            device=device
        )
        
        # build pose estimator
        pose_estimator = init_pose_estimator(
            pose_config,
            pose_checkpoint,
            device=device,
            cfg_options=cfg_options
        )

        return detector, pose_estimator

    def frame_inference(self, frame):
        # Run inference on single frame
        # MMPose tutorial: https://colab.research.google.com/drive/1rrCq6uPq6MhbNoslvBLqhljVKhqUO3RB#scrollTo=JjTt4LZAx_lK

        scope = self.detector.cfg.get('default_scope', 'mmdet')
        if scope is not None:
            init_default_scope(scope)

        start = time.time()
        detect_result = inference_detector(self.detector, frame)
        print(f"Inf det time = {time.time()-start}")

        pred_instance = detect_result.pred_instances.cpu().numpy()

        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                    pred_instance.scores > 0.3)]
        bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

        start = time.time()
        # predict keypoints
        pose_results = inference_topdown(self.pose_estimator, frame, bboxes)
        data_samples = merge_data_samples(pose_results)
        print(f"Pose inf time = {time.time()-start}")

        return data_samples

    def predict(self, file_path, conf=0):
        # Performs framewise pose estimation on video

        # Load video file
        cap = cv2.VideoCapture(file_path)
        keypoints_results = []
        pred_conf = []

        # Get video dimensions for normalisation later
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_i = 1

        # Progress bar
        with alive_bar(num_frames, title="..."+str(file_path[-23:-4]), length=16) as bar:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB for MMPose
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Begin timer
                start = time.time()

                # Run inference
                result = self.frame_inference(frame)

                # Display console message at each frame
                end = time.time()
                #print(f"{self.model_type} prediction for frame {frame_i} ({end-start} s)")
                frame_i += 1

                # Extract keypoints & confidence scores
                if result is not None:
                    # Take first prediction
                    res = result.pred_instances[0]

                    # Get keypoints and scores (normalised to the frame)
                    keypoints = res.keypoints[0]
                    normalised_kp = keypoints / [width, height]
                    scores = res.keypoint_scores[0]

                    keypoints_results.append(normalised_kp)
                    pred_conf.append(scores)
                else:
                    keypoints_results.append([[np.nan, np.nan] for i in range(len(self.KEYPOINT_DICT))])
                    pred_conf.append([np.nan for i in range(len(self.KEYPOINT_DICT))])

                bar()

        cap.release()

        # Keypoints x and y need to be swapped to be consistent with other models
        keypoints_results, pred_conf = reposition_keypoints(
                keypoints_results,
                pred_conf,
                swap_xy = True
            )
        
        return keypoints_results, pred_conf