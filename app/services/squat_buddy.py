import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow_docs.vis import embed
import numpy as np
import cv2
from tqdm import tqdm
import os
import joblib

# Import matplotlib libraries
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

from .constants import KEYPOINT_EDGE_INDS_TO_COLOR, SAMPLING_RATE, KEYPOINT_DICT
from .angle import *

input_size = 192

class SquatBuddy:

    def __init__(self):
        self.model = tf.lite.Interpreter(model_path='app/services/model.tflite')
        self.model.allocate_tensors()
        self.classifier = joblib.load('app/services/random_forest_model_mcc_02072025.pkl')
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
    
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
        model = self.module.signatures['serving_default']

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores
    
    def _keypoints_and_edges_for_display(self, keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
        """Returns high confidence keypoints and edges for visualization.

        Args:
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.
            height: height of the image in pixels.
            width: width of the image in pixels.
            keypoint_threshold: minimum confidence score for a keypoint to be
            visualized.

        Returns:
            A (keypoints_xy, edges_xy, edge_colors) containing:
            * the coordinates of all keypoints of all detected entities;
            * the coordinates of all skeleton edges of all detected entities;
            * the colors in which the edges should be plotted.
        """
        keypoints_all = []
        keypoint_edges_all = []
        edge_colors = []
        num_instances, _, _, _ = keypoints_with_scores.shape
        for idx in range(num_instances):
            kpts_x = keypoints_with_scores[0, idx, :, 1]
            kpts_y = keypoints_with_scores[0, idx, :, 0]
            kpts_scores = keypoints_with_scores[0, idx, :, 2]
            kpts_absolute_xy = np.stack(
                [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
            kpts_above_thresh_absolute = kpts_absolute_xy[
                kpts_scores > keypoint_threshold, :]
            keypoints_all.append(kpts_above_thresh_absolute)

            for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
                if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                    kpts_scores[edge_pair[1]] > keypoint_threshold):
                    x_start = kpts_absolute_xy[edge_pair[0], 0]
                    y_start = kpts_absolute_xy[edge_pair[0], 1]
                    x_end = kpts_absolute_xy[edge_pair[1], 0]
                    y_end = kpts_absolute_xy[edge_pair[1], 1]
                    line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                    keypoint_edges_all.append(line_seg)
                    edge_colors.append(color)
        if keypoints_all:
            keypoints_xy = np.concatenate(keypoints_all, axis=0)
        else:
            keypoints_xy = np.zeros((0, 17, 2))

        if keypoint_edges_all:
            edges_xy = np.stack(keypoint_edges_all, axis=0)
        else:
            edges_xy = np.zeros((0, 2, 2))
        return keypoints_xy, edges_xy, edge_colors

    def draw_prediction_on_image(
        self, image, keypoints_with_scores, crop_region=None, close_figure=False,
        output_image_height=None):
        """Draws the keypoint predictions on image.

        Args:
            image: A numpy array with shape [height, width, channel] representing the
            pixel values of the input image.
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.
            crop_region: A dictionary that defines the coordinates of the bounding box
            of the crop region in normalized coordinates (see the init_crop_region
            function below for more detail). If provided, this function will also
            draw the bounding box on the image.
            output_image_height: An integer indicating the height of the output image.
            Note that the image aspect ratio will be the same as the input image.

        Returns:
            A numpy array with shape [out_height, out_width, channel] representing the
            image overlaid with keypoint predictions.
        """
        height, width, channel = image.shape
        aspect_ratio = float(width) / height
        fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
        # To remove the huge white borders
        fig.tight_layout(pad=0)
        ax.margins(0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.axis('off')

        im = ax.imshow(image)
        line_segments = LineCollection([], linewidths=(2), linestyle='solid')
        ax.add_collection(line_segments)
        # Turn off tick labels
        scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

        (keypoint_locs, keypoint_edges,
        edge_colors) = self._keypoints_and_edges_for_display(
            keypoints_with_scores, height, width)

        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
        if keypoint_edges.shape[0]:
            line_segments.set_segments(keypoint_edges)
            line_segments.set_color(edge_colors)
        if keypoint_locs.shape[0]:
            scat.set_offsets(keypoint_locs)

        if crop_region is not None:
            xmin = max(crop_region['x_min'] * width, 0.0)
            ymin = max(crop_region['y_min'] * height, 0.0)
            rec_width = min(crop_region['x_max'], 0.99) * width - xmin
            rec_height = min(crop_region['y_max'], 0.99) * height - ymin
            rect = patches.Rectangle(
                (xmin,ymin),rec_width,rec_height,
                linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        if output_image_height is not None:
            output_image_width = int(output_image_height / height * width)
            image_from_plot = cv2.resize(
                image_from_plot, dsize=(output_image_width, output_image_height),
                interpolation=cv2.INTER_CUBIC)
        return image_from_plot
    
    def draw_prediction_on_empty_image(
        self, image, keypoints_with_scores, crop_region=None, close_figure=False,
        output_image_height=None):
        """Draws the keypoint predictions on image.

        Args:
            image: A numpy array with shape [height, width, channel] representing the
            pixel values of the input image.
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.
            crop_region: A dictionary that defines the coordinates of the bounding box
            of the crop region in normalized coordinates (see the init_crop_region
            function below for more detail). If provided, this function will also
            draw the bounding box on the image.
            output_image_height: An integer indicating the height of the output image.
            Note that the image aspect ratio will be the same as the input image.

        Returns:
            A numpy array with shape [out_height, out_width, channel] representing the
            image overlaid with keypoint predictions.
        """
        height, width, channel = image.shape
        aspect_ratio = float(width) / height
        fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
        # To remove the huge white borders
        fig.tight_layout(pad=0)
        ax.margins(0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.axis('off')

        im = ax.imshow(image)
        line_segments = LineCollection([], linewidths=(4), linestyle='solid')
        ax.add_collection(line_segments)
        # Turn off tick labels
        scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

        (keypoint_locs, keypoint_edges,
        edge_colors) = self._keypoints_and_edges_for_display(
            keypoints_with_scores, height, width)

        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
        if keypoint_edges.shape[0]:
            line_segments.set_segments(keypoint_edges)
            line_segments.set_color(edge_colors)
        if keypoint_locs.shape[0]:
            scat.set_offsets(keypoint_locs)

        if crop_region is not None:
            xmin = max(crop_region['x_min'] * width, 0.0)
            ymin = max(crop_region['y_min'] * height, 0.0)
            rec_width = min(crop_region['x_max'], 0.99) * width - xmin
            rec_height = min(crop_region['y_max'], 0.99) * height - ymin
            rect = patches.Rectangle(
                (xmin,ymin),rec_width,rec_height,
                linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        if output_image_height is not None:
            output_image_width = int(output_image_height / height * width)
            image_from_plot = cv2.resize(
                image_from_plot, dsize=(output_image_width, output_image_height),
                interpolation=cv2.INTER_CUBIC)
        return image_from_plot

    def to_gif(self, images, duration):
        """Converts image sequence (4D numpy array) to gif."""
        imageio.mimsave('./animation.gif', images, duration=duration)
        return embed.embed_file('./animation.gif')

    def progress(self, value, max=100):
        return HTML("""
            <progress
                value='{value}'
                max='{max}',
                style='width: 100%'
            >
                {value}
            </progress>
        """.format(value=value, max=max))
    
    def predict(self, image):
        
        # extract keypoints
        kpwv = self._process_image(image)

        # extract results for classification:
        results = self._format_result(kpwv[0][0])
        input = np.array([v for v in results.values()]).reshape(1, -1)

        # predict
        prediction = self.classifier.predict(input)

        # convert to appropriate datatype:
        prediction_list = prediction.tolist()

        return prediction_list
    
    def process_image(self, path: str):
        """
        Test running inference on one image
        """

        image = cv2.imread(path)

        img = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192,192)
        input_image = tf.cast(img, dtype=tf.int32)

        # Run model inference.
        self.model.set_tensor(self.input_details[0]['index'], np.array(input_image))
        self.model.invoke()
        keypoints_with_scores = self.model.get_tensor(self.output_details[0]['index'])

        return keypoints_with_scores
    
    def _process_image(self, image):
        img = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192,192)
        input_image = tf.cast(img, dtype=tf.int32)

        # Run model inference.
        self.model.set_tensor(self.input_details[0]['index'], np.array(input_image))
        self.model.invoke()
        keypoints_with_scores = self.model.get_tensor(self.output_details[0]['index'])

        return keypoints_with_scores
    
    def _correct_rotation(self, image, path):
        """Rotates the image if the video contains EXIF rotation metadata."""
        vidcap = cv2.VideoCapture(path)
        rotate_code = None

        rotation_flag = vidcap.get(cv2.CAP_PROP_ORIENTATION_META)

        if rotation_flag == 90:
            rotate_code = cv2.ROTATE_90_CLOCKWISE
        elif rotation_flag == 180:
            rotate_code = cv2.ROTATE_180
        elif rotation_flag == 270:
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

        if rotate_code is not None:
            image = cv2.rotate(image, rotate_code)

        return image
    
    def _video_to_frame(self, path: str, output_dir: str):
        """
        Turns a video into its individual frames and saves them to a directory.

        Args:
            path (str): path to video
            dir (Str): dir to save all images to
        """
        print('starting eval')

        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0

        frame_name = path.split(".")[0].split("/")[-1]

        labels = []

        while success:
            frame_path = os.path.join(output_dir, f"{frame_name}_frame{count:04d}.jpg")
            success, image = vidcap.read()
            image = self._correct_rotation(image, path)
            try:
                kpwv = self._process_image(image)
                results = self._format_result(kpwv[0][0])
                labels.append(
                    {
                        "path": frame_path,
                        "results": results,
                        "labeled": False
                    }
                )
                image_with_keypoints = self.draw_prediction_on_image(image, kpwv)
                cv2.imwrite(frame_path, image_with_keypoints)
            
            except Exception as e:    
                print(e)
            
            count += 1
        
        vidcap.release()
        return labels
    
    def _write_projections_to_image(self, image, keypoints_with_scores):
        frame_size = (image.shape[1], image.shape[0])
        
        # Write keypoints onto image
        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
        output_overlay = self.draw_prediction_on_image(
            np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores
        )

        # Resize the output_overlay back to the original frame size
        output_overlay = cv2.resize(output_overlay, frame_size)
        
        # Ensure the output overlay is of type uint8
        output_overlay = output_overlay.astype(np.uint8)
        
        return output_overlay
    
    def convert_numpy_floats(self, obj):
        """
        Recursively converts numpy float types in a dictionary to Python native float types.
        """
        if isinstance(obj, dict):
            return {key: self.convert_numpy_floats(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_floats(item) for item in obj]
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)  # Convert to native Python float
        else:
            return obj
    
    def _format_result(self, keypoints_with_scores):
        
        result = {
            "left_shoulder_x": keypoints_with_scores[KEYPOINT_DICT['left_shoulder']][0],
            "left_shoulder_y": keypoints_with_scores[KEYPOINT_DICT['left_shoulder']][1],
            "left_shoulder_c": keypoints_with_scores[KEYPOINT_DICT['left_shoulder']][2],
            "right_shoulder_x": keypoints_with_scores[KEYPOINT_DICT['right_shoulder']][0],
            "right_shoulder_y": keypoints_with_scores[KEYPOINT_DICT['right_shoulder']][1],
            "right_shoulder_c": keypoints_with_scores[KEYPOINT_DICT['right_shoulder']][2],
            "left_hip_x": keypoints_with_scores[KEYPOINT_DICT['left_hip']][0],
            "left_hip_y": keypoints_with_scores[KEYPOINT_DICT['left_hip']][1],
            "left_hip_c": keypoints_with_scores[KEYPOINT_DICT['left_hip']][2],
            "right_hip_x": keypoints_with_scores[KEYPOINT_DICT['right_hip']][0],
            "right_hip_y": keypoints_with_scores[KEYPOINT_DICT['right_hip']][1],
            "right_hip_c": keypoints_with_scores[KEYPOINT_DICT['right_hip']][2],
            "left_knee_x": keypoints_with_scores[KEYPOINT_DICT['left_knee']][0],
            "left_knee_y": keypoints_with_scores[KEYPOINT_DICT['left_knee']][1],
            "left_knee_c": keypoints_with_scores[KEYPOINT_DICT['left_knee']][2],
            "right_knee_x": keypoints_with_scores[KEYPOINT_DICT['right_knee']][0],
            "right_knee_y": keypoints_with_scores[KEYPOINT_DICT['right_knee']][1],
            "right_knee_c": keypoints_with_scores[KEYPOINT_DICT['right_knee']][2],
            "left_ankle_x": keypoints_with_scores[KEYPOINT_DICT['left_ankle']][0],
            "left_ankle_y": keypoints_with_scores[KEYPOINT_DICT['left_ankle']][1],
            "left_ankle_c": keypoints_with_scores[KEYPOINT_DICT['left_ankle']][2],
            "right_ankle_x": keypoints_with_scores[KEYPOINT_DICT['right_ankle']][0],
            "right_ankle_y": keypoints_with_scores[KEYPOINT_DICT['right_ankle']][1],
            "right_ankle_c": keypoints_with_scores[KEYPOINT_DICT['right_ankle']][2],
        }

        right_hip = np.array([result["right_hip_x"], result["right_hip_y"]])
        left_hip = np.array([result["left_hip_x"], result["left_hip_y"]])
        right_shoulder = np.array([result["right_shoulder_x"], result["right_shoulder_y"]])
        left_shoulder = np.array([result["left_shoulder_x"], result["left_shoulder_y"]])
        right_knee = np.array([result["right_knee_x"], result["right_knee_y"]])
        left_knee = np.array([result["left_knee_x"], result["left_knee_y"]])
        right_ankle = np.array([result["right_ankle_x"], result["right_ankle_y"]])
        left_ankle = np.array([result["left_ankle_x"], result["left_ankle_y"]])

        # Define vectors (Ensure they are NumPy arrays)
        r_a1 = np.array(right_hip) - np.array(right_shoulder)
        r_b1 = np.array(right_hip) - np.array(right_knee)
        r_a2 = np.array(right_knee) - np.array(right_hip)
        r_b2 = np.array(right_knee) - np.array(right_ankle)

        l_a1 = np.array(left_hip) - np.array(left_shoulder)
        l_b1 = np.array(left_hip) - np.array(left_knee)
        l_a2 = np.array(left_knee) - np.array(left_hip)
        l_b2 = np.array(left_knee) - np.array(left_ankle)

        # Compute angles (in radians)
        result["left_hip_angle"] = get_angle(l_a1, l_b1)
        result["right_hip_angle"] = get_angle(r_a1, r_b1)
        result["left_knee_angle"] = get_angle(l_a2, l_b2)
        result["right_knee_angle"] = get_angle(r_a2, r_b2)

        converted_result = self.convert_numpy_floats(result)

        return converted_result
