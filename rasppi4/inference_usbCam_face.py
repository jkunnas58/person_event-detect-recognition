#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

from os import X_OK
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import collections
import six
import PIL.Image as Image
import pyrealsense2 as rs
from utils import label_map_util
from utils import visualization_utils_color as vis_util
from send import SendData
from coordinate_converter import ConvertCoordinates
import copy


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
# PATH_TO_CKPT = r"C:\dev\tensorflow\workspace\tensorflow-face-detection-master\model\frozen_inference_graph_face.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'
# PATH_TO_LABELS = r"C:\dev\tensorflow\workspace\tensorflow-face-detection-master\protos\face_label_map.pbtxt"

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
                                        label_map,
                                        max_num_classes=NUM_CLASSES,
                                        use_display_name=True)
category_index = label_map_util.create_category_index(categories)


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
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(
                        class_name,
                        int(100*scores[i]))
                else:
                    display_str = 'score: {}%'.format(int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
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
            return False


class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        # print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)




if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 2:
#         print ("usage:%s (cameraID | filename) Detect faces\
#  in the video example:%s 0"%(sys.argv[0], sys.argv[0]))
#         exit(1)

#     try:
#     	camID = int(sys.argv[1])
#     except:
#     	camID = sys.argv[1]
   


    camID = 0
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
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    cap = cv2.VideoCapture(camID)
    windowNotSet = True

    #socket sending
    send_data_to_socket = SendData()
    send_data_to_socket.setup_server_sending()

    #Converterclass
    coordinate_converter = ConvertCoordinates()
    coordinate_converter.set_camera_resolution((640,480)) #camera resolution
    coordinate_converter.set_eye_center_offset_from_screen(-10) # distance to fictive eye center behind monitor
    coordinate_converter.set_mode('3D')
    # coordinate_converter.set_xyz(50,50,1000) #default point to look at top left looking at head
    depth_previous = 0.8

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

        image = color_image
        image_depth = depth_image

        # ret, image = cap.read()
        # if ret == 0:
        #     break

        [h, w] = image.shape[:2]
        # print (h, w)
        # image = cv2.flip(image, 1)
        # print(image.shape)
        (boxes, scores, classes, num_detections) = tDetector.run(image)

        # vis_util.visualize_boxes_and_labels_on_image_array(
        #     image,
        #     np.squeeze(boxes),
        #     np.squeeze(classes).astype(np.int32),
        #     np.squeeze(scores),
        #     category_index,
        #     use_normalized_coordinates=True,
        #     line_thickness=4)

        
        
        box_test = get_eye_focus_coordinate(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
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
            
            # 1/3 from the left of the box
            x_location = int(((box_int_list[1]-box_int_list[0])*1/3)+box_int_list[0]) 
            #1/3 from the top. 
            y_location = int(((box_int_list[3]-box_int_list[2])*1/3)+box_int_list[2])
            # get depth from realsense camera
            depth_location = depth_frame.get_distance(x_location, y_location) # depth in xx units
            depth_location_left = depth_frame.get_distance(x_location, y_location+10)
            depth_location_right = depth_frame.get_distance(x_location, y_location-10)
            depth_location = np.mean([depth_location,depth_location_right,depth_location_left])
            # Write some Text
            if depth_location < 0.01:
                depth_location = depth_previous

            depth_previous = copy.deepcopy(depth_location)

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (box_int_list[1],box_int_list[2])
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2

            cv2.putText(image,f'{round(depth_location,3)} meter', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

            
            circle_radius = 10
            circle_color = (0,0,255)
            circle_coordinates = (x_location,y_location)
            cv2.circle(image,circle_coordinates,circle_radius, circle_color, thickness=-1 ) 
            
            coordinate_converter.set_xyz(
                circle_coordinates[0],
                circle_coordinates[1],
                depth_location*1000
            )      

            try:
                str_data_to_send = coordinate_converter.get_eye_coordinates()
                # print(str_data_to_send)
                send_data_to_socket.send_data(str_data_to_send)
                time.sleep(0.05)
            except Exception:
                # str_data_to_send = coordinate_converter.get_eye_coordinates()
                # send_data_to_socket.send_data(str_data_to_send)
                time.sleep(0.05)

            # print(circle_coordinates[0], circle_coordinates[1])
            # print(f'depth to focus point {depth_location}')         
            # except:
            #     print('passed')
        else:            
            # print('no output')
            pass
        

        if windowNotSet is True:
            cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            windowNotSet = False

        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
