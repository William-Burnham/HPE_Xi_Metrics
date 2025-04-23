import tensorflow as tf # type:ignore
import tensorflow_hub as hub # type: ignore
from tensorflow_docs.vis import embed # type: ignore
import numpy as np
import cv2 # type: ignore
import os
import time
from alive_progress import alive_bar # type:ignore

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio # type:ignore
from IPython.display import HTML, display

from models.hpe_models.hpe_model import HPE_Model

# MoveNet: https://www.tensorflow.org/hub/tutorials/movenet
class MoveNet_Model(HPE_Model):
    def __init__(self, model_type='movenet_lightning'):
        HPE_Model.__init__(self)

        self.model_type = model_type

        if "movenet_lightning" in self.model_type:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            self.input_size = 192
        elif "movenet_thunder" in self.model_type:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            self.input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % self.model_type)

        self.model = module.signatures['serving_default']

        # Dictionary that maps from joint names to keypoint indices.
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

    def predict(self,  file_path, conf=0.1):

        if conf: self.conf = conf

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {file_path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pred_keypoints = []
        pred_conf = []
        frame_i = 0

        with alive_bar(num_frames, title="..."+str(file_path[-23:-4]), length=16) as bar:

            # Run inference on a framewise basis to optimise memory
            while True:
                ret, frame = cap.read()
                if not ret: break

                # Convert BGR (OpenCV format) to RGB
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.input_size, self.input_size))
                frame = tf.convert_to_tensor(frame, dtype=tf.float32)

                # run init_crop on first frame, otherwise determine from previous frame
                if frame_i == 0:
                    image_height, image_width, _ = frame.shape
                    crop_region = self.init_crop_region(image_height, image_width)
                else:
                    crop_region = self.determine_crop_region(keypoints_with_scores, image_height, image_width)

                start = time.time()
                # Make prediction
                keypoints_with_scores = self.run_inference(
                    self.movenet, frame, crop_region,
                    crop_size=[self.input_size, self.input_size]
                )
                end = time.time()

                pred_keypoints.append(np.ndarray.tolist(keypoints_with_scores[0, 0, :, :2]))
                pred_conf.append(np.ndarray.tolist(keypoints_with_scores[:, :, :, 2].flatten()))

                #print(f"{self.model_type} prediction for frame {frame_i} ({end-start} s)")
                frame_i += 1

                bar()

        cap.release()

        return pred_keypoints, pred_conf

    def movenet(self, input_image):
            """Runs detection on an input image.

            Args:
            input_image: A [1, height, width, 3] tensor represents the input image
                pixels. Note that the height/width should already be resized and match the
                expected input resolution of the model before passing into this function.

            Returns:
            A [1, 1, 17, 3] float numpy array representing the predicted keypoint
            coordinates and scores.
            """

            # SavedModel format expects tensor type of int32.
            input_image = tf.cast(input_image, dtype=tf.int32)
            # Run model inference.
            outputs = self.model(input_image)
            # Output is a [1, 1, 17, 3] tensor.
            keypoints_with_scores = outputs['output_0'].numpy()
            return keypoints_with_scores

    def init_crop_region(self, image_height, image_width):
        """Defines the default crop region.

        The function provides the initial crop region (pads the full image from both
        sides to make it a square image) when the algorithm cannot reliably determine
        the crop region from the previous frame.
        """
        if image_width > image_height:
            box_height = image_width / image_height
            box_width = 1.0
            y_min = (image_height / 2 - image_width / 2) / image_height
            x_min = 0.0
        else:
            box_height = 1.0
            box_width = image_height / image_width
            y_min = 0.0
            x_min = (image_width / 2 - image_height / 2) / image_width

        return {
            'y_min': y_min,
            'x_min': x_min,
            'y_max': y_min + box_height,
            'x_max': x_min + box_width,
            'height': box_height,
            'width': box_width
        }

    def torso_visible(self, keypoints):
        """Checks whether there are enough torso keypoints.

        This function checks whether the model is confident at predicting one of the
        shoulders/hips which is required to determine a good crop region.
        """
        return ((keypoints[0, 0, self.KEYPOINT_DICT['left_hip'], 2] >
                self.conf or
                keypoints[0, 0, self.KEYPOINT_DICT['right_hip'], 2] >
                self.conf) and
                (keypoints[0, 0, self.KEYPOINT_DICT['left_shoulder'], 2] >
                self.conf or
                keypoints[0, 0, self.KEYPOINT_DICT['right_shoulder'], 2] >
                self.conf))

    def determine_torso_and_body_range(
        self, keypoints, target_keypoints, center_y, center_x):
        """Calculates the maximum distance from each keypoints to the center location.

        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determineCropRegion for more detail.
        """
        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for joint in self.KEYPOINT_DICT.keys():
            if keypoints[0, 0, self.KEYPOINT_DICT[joint], 2] < self.conf:
                continue
            dist_y = abs(center_y - target_keypoints[joint][0]);
            dist_x = abs(center_x - target_keypoints[joint][1]);
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y

            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    def determine_crop_region(
        self, keypoints, image_height,
        image_width):
        """Determines the region to crop the image for the model to run inference on.

        The algorithm uses the detected joints from the previous frame to estimate
        the square region that encloses the full body of the target person and
        centers at the midpoint of two hip joints. The crop size is determined by
        the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions, the
        function returns a default crop which is the full image padded to square.
        """
        target_keypoints = {}
        for joint in self.KEYPOINT_DICT.keys():
            target_keypoints[joint] = [
            keypoints[0, 0, self.KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, self.KEYPOINT_DICT[joint], 1] * image_width
            ]

        if self.torso_visible(keypoints):
            center_y = (target_keypoints['left_hip'][0] +
                        target_keypoints['right_hip'][0]) / 2;
            center_x = (target_keypoints['left_hip'][1] +
                        target_keypoints['right_hip'][1]) / 2;

            (max_torso_yrange, max_torso_xrange,
            max_body_yrange, max_body_xrange) = self.determine_torso_and_body_range(
                keypoints, target_keypoints, center_y, center_x)

            crop_length_half = np.amax(
                [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
                max_body_yrange * 1.2, max_body_xrange * 1.2])

            tmp = np.array(
                [center_x, image_width - center_x, center_y, image_height - center_y])
            crop_length_half = np.amin(
                [crop_length_half, np.amax(tmp)]);

            crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

            if crop_length_half > max(image_width, image_height) / 2:
                return self.init_crop_region(image_height, image_width)
            else:
                crop_length = crop_length_half * 2;
                return {
                    'y_min': crop_corner[0] / image_height,
                    'x_min': crop_corner[1] / image_width,
                    'y_max': (crop_corner[0] + crop_length) / image_height,
                    'x_max': (crop_corner[1] + crop_length) / image_width,
                    'height': (crop_corner[0] + crop_length) / image_height -
                        crop_corner[0] / image_height,
                    'width': (crop_corner[1] + crop_length) / image_width -
                        crop_corner[1] / image_width
                }
        else:
            return self.init_crop_region(image_height, image_width)

    def crop_and_resize(self, image, crop_region, crop_size):
        """Crops and resize the image to prepare for the model input."""
        boxes=[[crop_region['y_min'], crop_region['x_min'],
                crop_region['y_max'], crop_region['x_max']]]
        output_image = tf.image.crop_and_resize(
            image, box_indices=[0], boxes=boxes, crop_size=crop_size)
        return output_image

    def run_inference(self, movenet, image, crop_region, crop_size):
        """Runs model inference on the cropped region.

        The function runs the model inference on the cropped region and updates the
        model output to the original image coordinate system.
        """
        image_height, image_width, _ = image.shape
        input_image = self.crop_and_resize(
            tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
        # Run model inference.
        keypoints_with_scores = movenet(input_image)
        # Update the coordinates.
        for idx in range(17):
            keypoints_with_scores[0, 0, idx, 0] = (
                crop_region['y_min'] * image_height +
                crop_region['height'] * image_height *
                keypoints_with_scores[0, 0, idx, 0]) / image_height
            keypoints_with_scores[0, 0, idx, 1] = (
                crop_region['x_min'] * image_width +
                crop_region['width'] * image_width *
                keypoints_with_scores[0, 0, idx, 1]) / image_width
        return keypoints_with_scores
    
    def get_model_type(self):
        return self.model_type
    
    def get_kpd(self):
        return self.KEYPOINT_DICT
    
    def get_connections(self):
        return self.CONNECTIONS
    
    def get_confidence(self):
        return self.conf