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
parser.add_argument('labelmap', help='Path to a labelmap txt format file.')
parser.add_argument('input_img', help='Path to an input image.')
parser.add_argument('output_img', help='Path to the output image.')
parser.add_argument('--score_threshold', help='Score threshold.', type=float, default=0.5)
parser.add_argument('--measure_predtime', help='Whether to measure prediction time', type=bool, default=False)

def get_session():
    # set tf backend to allow memory to grow, instead of claiming everything
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def load_labelmap(filepath):
    labelmap = {}
    with open(filepath, 'r') as f:
        classes = f.readlines()
        classes = [cls.strip() for cls in classes]
        for i in range(len(classes)):
            labelmap[i] = classes[i]
    return labelmap

def main(args):

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # ==========================================
    # Load pretrained model
    # ==========================================
    model = keras.models.load_model(os.path.expanduser(args.model), custom_objects=custom_objects)
    labelmap = load_labelmap(os.path.expanduser(args.labelmap))

    # ==========================================
    # Detect bounding boxes
    # ==========================================
    input_img = cv2.imread(os.path.expanduser(args.input_img))
    #input_img = read_image_bgr(os.path.expanduser(args.input_img))
    output_img = input_img.copy()
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    input_img = preprocess_image(input_img)
    input_img, scale = resize_image(input_img)
    input_img = np.expand_dims(input_img, axis=0)

    # detect bounding boxes
    _, _, boxes, nms_classification = model.predict_on_batch(input_img)
    num_detected_boxes = len(boxes[0, :, :])

    # measure prediction time
    if args.measure_predtime:
        times = []
        for i in range(100):
            stime = time.time()
            model.prediction_on_batch(input_img)
            etime = time.time()
            times.append(etime - stime)
        print('mean prediction time: %f [sec]' % np.mean(times))

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
