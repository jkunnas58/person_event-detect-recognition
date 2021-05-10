#!/usr/bin/env python
# coding: utf-8
"""
Detect Objects Using Intel Realsense
modification made by Jari Kunnas 2021
================================
"""


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
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')


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
# ~~~~~~~~~~~~~~~~~~~~~
import pyrealsense2 as rs
import numpy as np
import cv2


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)


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
try:

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # depth_colormap_dim = depth_colormap.shape
        # color_colormap_dim = color_image.shape

        # # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #     images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        image_np = color_image
        image_depth = depth_image


        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        image_np_expanded = np.tile(
            np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np_expanded, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        # net = cv2.dnn.readNetFromTensorflow(r"frozen_inference_graph.pb", 
                                    # r"faster_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        
        # print(boxes.shape[0])

        # viz_utils.visualize_boxes_and_labels_on_image_array(
        #     image_np_with_detections,
        #     detections['detection_boxes'][0].numpy(),
        #     (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        #     detections['detection_scores'][0].numpy(),
        #     category_index,
        #     use_normalized_coordinates=True,
        #     max_boxes_to_draw=200,
        #     min_score_thresh=.3,
        #     agnostic_mode=False)
        try:
            box_test = get_eye_focus_coordinate(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.3,
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

        print(f'depth image shape: {depth_image[1]}')
        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', cv2.resize(image_np_with_detections, (640, 480)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # cv2.waitKey(1)    



finally:
    # Stop streaming
    pipeline.stop()
    # cv2.destroyAllWindows()

# %%
