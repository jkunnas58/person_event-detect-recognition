#!/usr/bin/env python
# coding: utf-8
"""
Detect Objects Using Your Webcam
================================
"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" detection model to
# detect objects in the video stream extracted from your camera.

# %%
# Create the data directory
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# The snippet shown below will create the ``data`` directory where all our data will be stored. The
# code will create a directory structure as shown bellow:
#
# .. code-block:: bash
#
#     data
#     └── models
#
# where the ``models`` folder will will contain the downloaded models.
import os

DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)

# %%
# Download the model
# ~~~~~~~~~~~~~~~~~~
# The code snippet shown below is used to download the object detection model checkpoint file,
# as well as the labels file (.pbtxt) which contains a list of strings used to add the correct
# label to each detection (e.g. person).
#
# The particular detection algorithm we will use is the `SSD ResNet101 V1 FPN 640x640`. More
# models can be found in the `TensorFlow 2 Detection Model Zoo <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>`_.
# To use a different model you will need the URL name of the specific model. This can be done as
# follows:
#
# 1. Right click on the `Model name` of the model you would like to use;
# 2. Click on `Copy link address` to copy the download link of the model;
# 3. Paste the link in a text editor of your choice. You should observe a link similar to ``download.tensorflow.org/models/object_detection/tf2/YYYYYYYY/XXXXXXXXX.tar.gz``;
# 4. Copy the ``XXXXXXXXX`` part of the link and use it to replace the value of the ``MODEL_NAME`` variable in the code shown below;
# 5. Copy the ``YYYYYYYY`` part of the link and use it to replace the value of the ``MODEL_DATE`` variable in the code shown below.
#
# For example, the download link for the model used below is: ``download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz``
# import tarfile
# import urllib.request

# # Download and extract model
# MODEL_DATE = '20200711'
# MODEL_NAME = 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'
# MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
# MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
# MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
# PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
# PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
# PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
# if not os.path.exists(PATH_TO_CKPT):
#     print('Downloading model. This may take a while... ', end='')
#     urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
#     tar_file = tarfile.open(PATH_TO_MODEL_TAR)
#     tar_file.extractall(MODELS_DIR)
#     tar_file.close()
#     os.remove(PATH_TO_MODEL_TAR)
#     print('Done')

# # Download labels file
# LABEL_FILENAME = 'mscoco_label_map.pbtxt'
# LABELS_DOWNLOAD_BASE = \
#     'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
# PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
# if not os.path.exists(PATH_TO_LABELS):
#     print('Downloading label file... ', end='')
#     urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
#     print('Done')

#%%
#Load custom model
PATH_TO_CFG = r"C:\dev\tensorflow\workspace\training_demo\exported-models\my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\pipeline.config"
PATH_TO_CKPT = r"C:\dev\tensorflow\workspace\training_demo\exported-models\my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\checkpoint"
PATH_TO_LABELS = r"C:\dev\tensorflow\workspace\training_demo\exported-models\my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\label_map.pbtxt"
# %%
# Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

# %%
import collections
import six
import PIL.Image as Image
def get_eye_focus_coordinate(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    mask_alpha=.4,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
    """
    
    """
      # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    box_to_keypoint_scores_map = collections.defaultdict(list)
    box_to_track_ids_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if keypoint_scores is not None:
                box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
            if track_ids is not None:
                box_to_track_ids_map[box] = track_ids[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in six.viewkeys(category_index):
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = f'{round(100*scores[i])}'
                    else:
                        display_str = f'{display_str}: {round(100*scores[i])}'
                if not skip_track_ids and track_ids is not None:
                    if not display_str:
                        display_str = f'ID {track_ids[i]}'
                    else:
                        display_str = f'{display_str}: ID { track_ids[i]}'
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                elif track_ids is not None:                
                    box_to_color_map[box] =groundtruth_box_visualization_color
                else:
                    box_to_color_map[box] = groundtruth_box_visualization_color
    #Export location of box in relative or absolute coordinates
    image_for_size = Image.fromarray(np.uint8(image)).convert('RGB')
    im_width, im_height = image_for_size.size 
    ymin, xmin, ymax, xmax = box
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    return (left, right, top, bottom)


# %%
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# %%
# Define the video stream
# ~~~~~~~~~~~~~~~~~~~~~~~
# We will use `OpenCV <https://pypi.org/project/opencv-python/>`_ to capture the video stream
# generated by our webcam. For more information you can refer to the `OpenCV-Python Tutorials <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#capture-video-from-camera>`_


import cv2

cap = cv2.VideoCapture(0)


# Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# Here are some simple things to try out if you are curious:
#
# * Modify some of the input images and see if detection still works. Some simple things to try out here (just uncomment the relevant portions of code) include flipping the image horizontally, or converting to grayscale (note that we still expect the input image to have 3 channels).
# * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.
import numpy as np

while True:
    # Read frame from camera
    ret, image_np = cap.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    print(f'Image shape from opencv and webcam is: {image_np.shape}')
    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    
    # print(boxes.shape[0])

    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #       image_np_with_detections,
    #       detections['detection_boxes'][0].numpy(),
    #       (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
    #       detections['detection_scores'][0].numpy(),
    #       category_index,
    #       use_normalized_coordinates=True,
    #       max_boxes_to_draw=200,
    #       min_score_thresh=.3,
    #       agnostic_mode=False)
    try:
        box_test = get_eye_focus_coordinate(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.5,
            agnostic_mode=False)
        # print(box_test) #example (0.23469042778015137, 0.30845338106155396, 0.7406021952629089, 0.5217226147651672)
            
        if box_test:
            # print(box_test)
            box_int_list = [0,0,0,0]
            for i in range(4):                
                box_int_list[i] = int(box_test[i])            
            
            circle_radius = 10
            circle_color = (0,0,255)
            # 1/3 from the left of the box
            x_location = int(((box_int_list[1]-box_int_list[0])*1/3)+box_int_list[0]) 
            #1/3 from the top. 
            y_location = int(((box_int_list[3]-box_int_list[2])*1/3)+box_int_list[2])

            circle_coordinates = (x_location,y_location)
            cv2.circle(image_np_with_detections,circle_coordinates,circle_radius, circle_color, thickness=-1 )            
            # except:
            #     print('passed')
        else:            
            # print('no output')
            pass
    except:
        pass

    
    # Display output
    cv2.imshow('object detection', image_np_with_detections)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# %%
