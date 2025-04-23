from ultralytics import YOLO # type: ignore
import numpy as np
import csv
import os

from models.hpe_models.hpe_model import HPE_Model
from utils.kp_utils import reposition_keypoints

class Yolo_Model(HPE_Model):
    def __init__(self, model_version='yolo11n-pose.pt'):
        HPE_Model.__init__(self)

        self.model_type = "yolopose"

        # model_version: Default is yolo11n-pose.pt. Use 'yolo11s-pose.pt' or 'yolo11m-pose.pt' for larger models
        self.model_version = model_version
        # Load pretrained YOLOPose model
        self.model = YOLO(self.model_version)

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

    def predict(self,  file_path, conf=0.0):
        """
        Args:
            file_path (str): Path to the video file.
            conf (float): Confidence threshold for predictions.

        Returns:
            List[List[float]]: Extracted keypoints for each frame.
        """

        if conf: self.conf = conf

        # Predict results
        if file_path == 0: raise Exception("file_path cannot be 0")
        print(file_path)
        results = self.model.predict(source=file_path, save=False, show=False, conf=self.conf, verbose=False, stream=True)

        # Extract keypoint values from prediction results (only keypoints of box 0)
        keypoints = []
        confidences = []
        for frame_result in results:
            if len(frame_result.keypoints) > 0:
                try:
                    xyn = frame_result.keypoints[0].xyn[0].tolist() # Normalised keypoint positions
                    c = frame_result.keypoints[0].conf[0].tolist() # Confidence level per keypoint
                    
                    keypoints.append(xyn)
                    confidences.append(c)
                except:
                    keypoints.append([[np.nan, np.nan] for i in range(len(self.KEYPOINT_DICT))])
                    confidences.append([np.nan for i in range(len(self.KEYPOINT_DICT))])
            else:
                keypoints.append([[np.nan, np.nan] for i in range(len(self.KEYPOINT_DICT))])
                confidences.append([np.nan for i in range(len(self.KEYPOINT_DICT))])

        # Keypoints x and y need to be swapped to be consistent with other models
        keypoints, confidences = reposition_keypoints(
            keypoints,
            confidences,
            swap_xy = True
        )

        return keypoints, confidences