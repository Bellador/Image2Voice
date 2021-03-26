#!/usr/bin/env python
# coding: utf-8

# dependancies
import os
import cv2
import math
import tarfile
import numpy as np
import urllib.request
from collections import defaultdict
# import packages for color prediction
import webcolors
from colorthief import ColorThief
# import object detection dependancies
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
# import text to speech library
import pyttsx3


def load_model():
    WORKSPACES_DIR = 'C:/Users/mhartman/PycharmProjects/Img2Voice'
    NEW_WORKSPACE_DIR = os.path.join(WORKSPACES_DIR, workspace_name)
    # create new workspace
    if not os.path.exists(NEW_WORKSPACE_DIR):
        os.mkdir(NEW_WORKSPACE_DIR)

    # create directories for new project with model and data
    DATA_DIR = os.path.join(NEW_WORKSPACE_DIR, 'data')
    MODELS_DIR = os.path.join(DATA_DIR, 'pre-trained-models')
    for dir_ in [DATA_DIR, MODELS_DIR]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)
    print(f'[+] using project folder under ./Img2Voice/{workspace_name}')
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

    # Download and extract model
    MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
    MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
    MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
    PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
    PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
    PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
    if not os.path.exists(PATH_TO_CKPT):
        print('[*] downloading model. This may take a while... ', end='')
        urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
        tar_file = tarfile.open(PATH_TO_MODEL_TAR)
        tar_file.extractall(MODELS_DIR)
        tar_file.close()
        os.remove(PATH_TO_MODEL_TAR)
        print('[+] model download complete.')

    # Download labels file
    LABEL_FILENAME = 'mscoco_label_map.pbtxt'
    LABELS_DOWNLOAD_BASE = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
    if not os.path.exists(PATH_TO_LABELS):
        print('[*] downloading label file... ', end='')
        urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
        print('[+] label file download complete.')

    # %%
    # Load the model
    # ~~~~~~~~~~~~~~
    # Next we load the downloaded model

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging

    tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

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
    print('[+] model loaded and checkpoints restored.')

    return detection_model, PATH_TO_LABELS

def img_to_voice(image, detections, category_index, normalised_coordinates=True, label_id_offset=1):
    # webcolors adaption taken from https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python
    def closest_colour(requested_colour):
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    def get_colour_name(requested_colour):
        try:
            closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
        except ValueError:
            closest_name = closest_colour(requested_colour)
            actual_name = None

        if actual_name == None:
            return closest_name
        else:
            return actual_name

    print('[*] starting object to voice translation')
    (im_height, im_width, channels) = image.shape
    # print(channels)
    boxes = detections['detection_boxes'][0].numpy()
    classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
    scores = detections['detection_scores'][0].numpy()
    # dict to store all objects with their metadata that are over the score threshold
    object_dict = defaultdict(lambda: dict(class_name=str, color_name=str))

    for i in range(boxes.shape[0]):
        box = tuple(boxes[i].tolist())
        ymin, xmin, ymax, xmax = box
        class_name = category_index[classes[i]]['name']
        score = scores[i]
        if score > min_acc_to_be_added:
            if normalised_coordinates:
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
            # shrink bbox to zoom in further on object to get a more accurat color prediction
            left = left + 0.2*(right - left)
            right = right - 0.2*(right - left)
            bottom = bottom + 0.2*(top - bottom)
            top = top - 0.1*(top - bottom)
            # round bbox boundaries to next pixel
            left = math.ceil(left)
            right = math.ceil(right)
            top = math.ceil(top)
            bottom = math.ceil(bottom)
            # extract pixel values from image where the bounding box is
            bbox_image = image[top:bottom, left:right]
            # store bbox to disk to analyse with ColorThief (easier)
            cv2.imwrite('./images/bbox_temp.jpg', bbox_image)
            # bbox_image = cv2.imread('./images/bbox_temp.jpg')
            color_thief = ColorThief('./images/bbox_temp.jpg')
            detected_rgb = color_thief.get_color(quality=1)
            # translate rgb value to english color names
            color_name = get_colour_name(detected_rgb)
            # store it in the object_dict for now
            object_dict[i+1]['class_name'] = class_name
            object_dict[i+1]['color_name'] = color_name

    print(f'[*] found {len(object_dict.keys())} objects')
    print(f'[+] constructing text representation of the object')

    text_object_description = 'The image contains '

    # check if only one object is present
    print(f'OBJECT DICT: {object_dict}')
    print(f'OBJECT DICT len: {len(object_dict.keys())}')
    if len(object_dict.keys()) == 1:
        dict_key = list(object_dict.keys())[0]
        color_name = object_dict[dict_key]['color_name']
        class_name = object_dict[dict_key]['class_name']
        text_object_description = f'The image contains a {color_name} colored {class_name}'
    # check if any objects above the threshold where detected
    elif len(object_dict.keys()) > 0:
        for index, object in enumerate(object_dict.values(), 0):
            if index < len(object_dict.keys()):
                color_name = object['color_name']
                class_name = object['class_name']
                text_object_description += f'a {color_name} colored {class_name} and '
            else:
                text_object_description += f'lastly also a {color_name} colored {class_name}.'
    else:
        text_object_description = 'The image contains no objects that the model knows.'
    # initiate text to voice object
    print(f'[*] intitialsing text to voice engine')
    engine = pyttsx3.init()
    # modify the speaking rate - make it slower
    speech_rate = engine.getProperty('rate')  # getting details of current speaking rate
    # print(speech_rate)  # printing current voice rate
    engine.setProperty('rate', speech_rate * 0.8)  # setting up new voice rate
    # feed it to text to speech engine
    print(f'[+] speaking...')
    print(f'[*] text: {text_object_description}')
    engine.say(text_object_description)
    print(f'[+] done speaking.')
    engine.runAndWait()
    del engine

def detection(detection_model, PATH_TO_LABELS, default_img='./images/broccoli.jpg'):

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    while True:
        # request user input where he can specify a custom image path besides the default
        image_filename = input('[*] enter image filename (with extension) that is stored under Img2Voice/images [ENTER for default image]\n')
        IMAGE_PATH = f'./images/{image_filename}'
        # check if path is valid, else take default
        if not os.path.isfile(IMAGE_PATH):
            print(f'[!] default image is used: {DEFAULT_IMAGE_PATH}')
            IMAGE_PATH = default_img
        print('[*] starting object detection')
        image_np = cv2.imread(IMAGE_PATH)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        # Display output
        cv2.destroyAllWindows()
        # prepare audio representation
        img_to_voice(image_np_with_detections, detections, category_index)
        # after img to voice add bounding box
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=min_acc_to_be_added,  # .60
            agnostic_mode=False)
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
        # removing temp file
        os.remove('./images/bbox_temp.jpg')
        # break on key
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    '''
    adjustable parameters below
    '''
    print('[*] Starting')
    # a project folder with the workspace name will be created in which the specified model will be stored
    workspace_name = 'workspace'
    # check TensorFlow 2 Detection Model Zoo for most recent models
    # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    MODEL_DATE = '20200711'
    MODEL_NAME = 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'
    DEFAULT_IMAGE_PATH = './images/broccoli.jpg' # can also be added later on when the model was loaded via the console
    min_acc_to_be_added = 0.5 # prediction accuracy to be added to the image and text 2 speech
    prediction_model, PATH_TO_LABELS = load_model()
    detection(prediction_model, PATH_TO_LABELS, default_img=DEFAULT_IMAGE_PATH)
    print('[*] Done.')
