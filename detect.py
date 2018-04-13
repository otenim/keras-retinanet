from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import keras
import argparse
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model', help='Path to a serialized model file.')
parser.add_argument('classes', help='Path to a classes file.')
parser.add_argument('input_img', help='Path to an input image.')
parser.add_argument('output_img', help='Path to the output image.')
parser.add_argument('--score_threshold', help='Score threshold.', type=float, default=0.5)

def load_classes(filepath):
    classes = {}
    with open(filepath, 'r') as f:
        class_names = f.readlines()
        class_names = [name.strip() for name in class_names]
        for i in range(len(class_names)):
            classes[i] = class_names[i]
    return classes

def main(args):

    # ==========================================
    # Load pretrained model
    # ==========================================
    model = keras.models.load_model(os.path.expanduser(args.model), custom_objects=custom_objects)
    labelmap = load_classes(os.path.expanduser(args.labelmap))

    # ==========================================
    # Detect bounding boxes
    # ==========================================
    input_img = cv2.imread(os.path.expanduser(args.input_img))
    output_img = input_img.copy()

    # preprocess image for network
    input_img = preprocess_image(input_img)
    input_img, scale = resize_image(input_img)
    input_img = np.expand_dims(input_img, axis=0)

    # detect bounding boxes
    _, _, boxes, nms_classification = model.predict_on_batch(input_img)
    num_detected_boxes = len(boxes[0, :, :])

    # visualize
    for i in range(num_detected_boxes):
        label = np.argmax(nms_classification[0, i, :])
        score = nms_classification[0, i, label]
        if score < args.score_threshold:
            continue

        # draw bounding box on a copy of the original input image
        color = label_color(label)
        coord = boxes[0,i,:] / scale
        draw_box(output_img, coord,color=color)

        # draw caption for the above box
        caption = '%s %.3f' % (labelmap[label], score)
        draw_caption(output_img, coord, caption=caption)

    # save output image
    cv2.imwrite(os.path.expanduser(args.output_img), output_img)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
