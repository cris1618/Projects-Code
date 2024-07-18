#!/mnt/raid1/pietro/home2/environment/bin/python3.10
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, ReLU
import tensorflow_datasets as tfds
import numpy as np
import random
from sklearn.model_selection import train_test_split
#import cv2
from matplotlib.ticker import MaxNLocator
#from keras.utils.layer_utils import count_params
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from collections import Counter
import zipfile
import os
import re
import xml.etree.ElementTree as ET
import itertools
from sklearn import preprocessing
import json
import csv
import pickle


def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    #print(boxes1.shape, boxes2.shape)
    #print(boxes1_corners.shape, boxes2_corners.shape)
    #quit()
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def visualize_detections(image, boxes, classes, scores, boxes_orig=None, cls_orig=None, unique_classes=None, colors_classes=None, figsize=(7, 7), linewidth=1):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    #index_classes = {val: i for i, val in enumerate(unique_classes)}
    #colors = [colors_classes[index_classes[val]] for val in classes]
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    #f, (ax, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    ax = plt.gca()
    color_temp = [0,0,1]
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=color_temp, linewidth=linewidth)
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": 'green', "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    if boxes_orig is not None:
        for box, _cls in zip(boxes_orig, cls_orig):
            text = "{}".format(_cls)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor='green', linewidth=linewidth)
            ax.add_patch(patch)
            ax.text(
                x1,
                y1,
                text,
                bbox={"facecolor": 'green', "alpha": 0.4},
                clip_box=ax.clipbox,
                clip_on=True,
            )
    #for n in range(len(unique_classes)):
     #   ax1.text()
    #plt.show()
    #plt.savefig('%s_prediction.png' % (file))
    return ax


def read_image(file):
    file = './CROPPED/'+file
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img, channels=3)
    #img = tf.image.convert_image_dtype(img, tf.float32)
    #print(file, tf.shape(img))
    return img.numpy()


def shuffle_generator(files):
    idx = np.arange(len(files))
    np.random.default_rng().shuffle(idx)
    for i in idx:
        image = read_image(files[i])
        #sample = {'image':image, 'image/file':files[i], 'bbox':boxes[i], 'label':labels[i]}
        yield image#, labels[i]


def data_to_dict(files, boxes_imgs, labels_imgs):
    le = preprocessing.LabelEncoder()
    le.fit(list(itertools.chain(*labels_imgs)))
    boxes_all = []
    labels_all = []
    for n, file in enumerate(files):
        #print(np.array(boxes_imgs[n]).shape, np.array(labels_imgs[n]).shape)
        boxes = np.array(boxes_imgs[n])
        labels = le.transform(np.array(labels_imgs[n]))
        """boxes[:, 0] = boxes[:, 0]/image.shape[1]
        boxes[:, 1] = boxes[:, 1]/image.shape[0]
        boxes[:, 2] = boxes[:, 2]/image.shape[1]
        boxes[:, 3] = boxes[:, 3]/image.shape[0]"""
        #annotation = {'bbox': boxes}
        #annotations.append(annotation)
        #boxes = tf.convert_to_tensor(boxes)
        #labels = tf.convert_to_tensor(labels)
        boxes_all.append(boxes)
        labels_all.append(labels)

    metadata = {'image': [], 'image/file': [], 'bbox': [], 'label': []}
    dataset = tf.data.Dataset.from_generator(shuffle_generator,
                                             args = [files],
                                             output_signature=(tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32)))
                                                               #tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                               #tf.TensorSpec(shape=(None,), dtype=tf.int32)))

    return dataset, le


def data_to_dict(image, boxes, labels, index):
    return {"image": image, "image_id": tf.convert_to_tensor(index),
            "annotations": {"bbox": tf.convert_to_tensor(boxes), "label": tf.convert_to_tensor(labels)}}


def id_to_data(index, df):
    index = index.numpy()
    file = df[df['img_id']==index].iloc[0]['file']
    image = read_image(file)
    boxes = df[df['img_id']==index]['box'].apply(lambda x: x.replace('[', '').replace(']','')).values
    boxes = np.asarray([[float(el) for el in box.split(' ') if len(el) > 0] for box in boxes], dtype=np.float32)
    labels = np.asarray(df[df['img_id']==index]['label'], dtype=np.int32)
    #image, boxes, labels = preprocess_solace(image, boxes, labels)
    return (image, boxes, labels)


def map_id_to_data(index, df):
    return tf.py_function(lambda x: id_to_data(x, df), inp=[index], Tout=[tf.float32, tf.float32, tf.int32])


def csv_to_dataset(df, indices):
    dataset = tf.data.Dataset.from_tensor_slices(indices)
    dataset = dataset.map(lambda x: map_id_to_data(x, df))
    dataset = dataset.shuffle(len(indices))
    return dataset


def data_to_csv(files, boxes_imgs, labels_imgs):
    """for labels in labels_imgs: #group all zp labels
        for n, label in enumerate(labels):
            if 'zp' in label:
                labels[n] = 'zp'
            if label == 'rhiz':
                labels[n] = 'for'"""
    le = preprocessing.LabelEncoder()
    le.fit(list(itertools.chain(*labels_imgs)))
    header = ['img_id', 'file', 'box_id', 'box', 'class', 'label']
    with open('solace.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for n, file in enumerate(files):
            #image = read_image(file)
            boxes = np.array(boxes_imgs[n])
            """boxes[:, 0] = boxes[:, 0] / image.shape[1]
            boxes[:, 1] = boxes[:, 1] / image.shape[0]
            boxes[:, 2] = boxes[:, 2] / image.shape[1]
            boxes[:, 3] = boxes[:, 3] / image.shape[0]"""
            labels = le.transform(np.array(labels_imgs[n]))
            for m in range(len(boxes)):
                if boxes[m][0] >= 176 and boxes[m][0] <= 5471-176 and boxes[m][2] >= 176 and boxes[m][2] <= 5471-176:
                    data = [n, file, str(n)+'-'+str(m), boxes[m], labels_imgs[n][m], labels[m]]
                    writer.writerow(data)


def get_boxes_labels(file_path):
    boxes = []
    labels = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for obj in root.findall('object')[:-1]:
        label = obj.find('name').text
        label = ''.join(label.split()).lower()
        if label == 'zphc':
            label = 'zpcop'
        if label == 'fpsalps':
            label = 'fpsalp'
        if label == 'fps':
            label = 'fpsmall'
        if label == 'aggfpl':
            label = 'aggfp'
        if label != 'zpsalp' and label != 'zpeu' and label != 'zp' and label != 'notclassified':
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

    return boxes, labels


def load_images(dir):
    #images = []
    boxes_imgs = []
    labels_imgs = []
    imgs = []
    filelist = []
    for root, dirs, files in os.walk(dir):
        if not ('5' in root or '6' in root): # disregard Stations 5 and 6 for the time being
            for file in files:
                path = os.path.join(root,file)
                path = os.path.relpath(path, os.getcwdb().decode('utf-8'))
                filelist.append(path)
    #for name in filelist:
        #print(name)

    for filename in filelist:#os.listdir(dir):
        if filename.endswith('.JPG'):
            #image_path = os.path.join(dir, filename)
            xml_path = filename[:-4] + '.xml'#os.path.join(dir, filename[:-4] + '.xml')
            #image = cv2.imread(image_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #images.append(image)
            #print(image.shape)
            boxes, labels = get_boxes_labels(xml_path)
            boxes_imgs.append(boxes)
            labels_imgs.append(labels)
            imgs.append(filename)
            #img = tf.io.read_file(filename)
            #img = tf.image.decode_jpeg(img, channels=3)
            #print(tf.shape(img))
    return imgs, boxes_imgs, labels_imgs


def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio


def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


def preprocess_solace(image, bbox, class_id):
    #image = sample["image"]
    #bbox = sample["annotations"]["bbox"]
    #class_id = tf.cast(sample["annotations"]["label"], dtype=tf.int32)

    #image, image_shape, _ = resize_and_pad_image(image)
    #print(image_shape)
    #print(image.shape)
    """bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )"""
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


def map_preprocess_solace(sample):
    return tf.py_function(preprocess_solace,
                          inp=[sample], Tout=[tf.uint8, tf.float32, tf.int32])


class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        #print(dims.numpy().shape)
        #print(centers)
        #quit()
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)


class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        #tf.print(tf.math.count_nonzero(matched_gt_idx))
        label = tf.concat([box_target, cls_target], axis=-1)
        return label


    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()


def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = tf.keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return tf.keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


class FeaturePyramid(tf.keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, backbone=None, **kwargs):
        super().__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = Conv2D(256, 3, 2, "same")
        self.upsample_2x = UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = tf.keras.Sequential([Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(Conv2D(256, 3, padding="same", kernel_initializer=kernel_init))
        head.add(ReLU())
    head.add(Conv2D(output_filters,3,1,padding="same",kernel_initializer=kernel_init, bias_initializer=bias_init))
    return head


class RetinaNet(tf.keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super().__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
            #tf.print(tf.shape(image), tf.shape(self.cls_head(feature)))
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        #tf.print(tf.shape(cls_outputs))
        return tf.concat([box_outputs, cls_outputs], axis=-1)


class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes= 18,
        confidence_threshold=0.5,
        nms_iou_threshold=0.5,
        max_detections_per_class=500,
        max_detections=1000,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super().__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super().__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes=18, alpha=0.25, gamma=2.0, delta=1.0):
        super().__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        #tf.print(tf.math.count_nonzero(positive_mask))
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        #tf.print(tf.shape(clf_loss), tf.shape(box_loss))
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        #tf.print(normalizer)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        #tf.print(clf_loss, box_loss)
        #for el in y_true[:, :, 4]:
            #tf.print(tf.shape(el))
        #tf.print(len(cls_predictions[0]))
        return loss


def export_classes(df):
    classes = df['class'].unique()
    class_dict = {}
    for element in classes:
        class_dict[element] = df[df['class'] == element].iloc[0]['label']
    num_classes = len(classes)
    """sns.countplot(x="class", data=df, palette="Set1")
    plt.title("class")
    plt.show()"""
    print(df['class'].value_counts())
    return num_classes, class_dict


def crop_image(filename, dir, df):
    filename_last = filename.split('/')[-1]
    path = filename#path = os.path.join(dir, filename)
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    boxes, boxes_id, labels = boxes_labels_from_csv(df, filename)
    cropped_list = []
    rejected_list = []
    names_list = []
    names_rejected = []
    hw_list = []
    boxes_list = []
    boxesid_list = []
    labels_list = []
    n_list = []
    n = 0
    #print(img.shape[0], img.shape[1])
    for h, offset_height in enumerate(range(0, img.shape[0] - 1023, 512)):
        for w, offset_width in enumerate(range(176, img.shape[1] - 1023, 512)):
            cropped = tf.image.crop_to_bounding_box(
                image=img,
                offset_height=offset_height,
                offset_width=offset_width,
                target_height=1024,
                target_width=1024
            )
            #cropped = tf.image.convert_image_dtype(cropped, tf.float32)
            mask = (boxes[:,0]>=offset_width) & (boxes[:,0]<=offset_width+1023) & (boxes[:,1] >= offset_height) \
                   & (boxes[:,1] <= offset_height+1023) & (boxes[:,2] <= offset_width+1023) & (boxes[:,3] <= offset_height+1023)
            boxes_crop = boxes[mask, :]
            #print(n, w, offset_width, offset_width + 1024, h, offset_height, offset_height + 1024)
            #print(boxes_crop)
            labels_crop = labels[mask]
            boxesid_crop = boxes_id[mask]
            if len(boxes_crop):
                cropped_list.append(cropped)
                names_list.append(filename_last[:-4]+'_CROPPED_'+str(n)+'.JPG')
                hw_list.append([h, w])
                boxes_crop[:,0] = boxes_crop[:,0] - w*512 - 176
                boxes_crop[:,1] = boxes_crop[:,1] - h*512
                boxes_crop[:,2] = boxes_crop[:,2] - w*512 - 176
                boxes_crop[:,3] = boxes_crop[:,3] - h*512
                boxes_list.append(boxes_crop)
                boxesid_list.append(boxesid_crop)
                labels_list.append(labels_crop)
                n_list.append(n)
            else:
                rejected_list.append(cropped)
                names_rejected.append(filename_last[:-4]+'_CROPPED_'+str(n)+'.JPG')
            n += 1
    #quit()
    return cropped_list, names_list, hw_list, boxes_list, boxesid_list, labels_list, n_list, rejected_list, names_rejected



def boxes_labels_from_csv(df, file):
    boxes = df[df['file'] == file]['box'].apply(lambda x: x.replace('[', '').replace(']', '')).values
    boxes = np.asarray([[float(el) for el in box.split(' ') if len(el) > 0] for box in boxes], dtype=np.int32)
    labels = np.asarray(df[df['file'] == file]['label'], dtype=np.int32)
    boxes_id = np.asarray(df[df['file'] == file]['box_id'])
    return boxes, boxes_id, labels


def check_rejected_box(box, filename, dir):
    path = filename#os.path.join(dir, filename)
    filename_last = filename.split('/')[-1]
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    x = (box[0] + box[2])/2
    y = (box[1] + box[3])/2
    area = (box[2]-box[0])*(box[3]-box[1])
    max_inters = 0
    n = 0
    n_max = 0
    h_max = w_max = 0
    for h, offset_height in enumerate(range(0, img.shape[0] - 1023, 512)):
        for w, offset_width in enumerate(range(176, img.shape[1] - 1023, 512)):
            if x >= offset_width and x <= offset_width + 1023 and y >= offset_height and y <= offset_height + 1023:
                dx = min(offset_width+1023, box[2]) - max(offset_width, box[0])
                dy = min(offset_height+1023, box[3]) - max(offset_height, box[1])
                inters_area = dx*dy
                if inters_area >= max_inters:
                    max_inters = inters_area
                    n_max = n
                    h_max = h
                    w_max = w
            n += 1
    new_box = np.zeros_like(box)
    if box[0] < w_max*512+176:
        new_box[0] = 0
    else:
        new_box[0] = box[0] - w_max*512-176
    if box[2] > w_max*512+176+1023:
        new_box[2] = 1023
    else:
        new_box[2] = box[2] - w_max*512-176
    if box[1] < h_max*512:
        new_box[1] = 0
    else:
        new_box[1] = box[1] - h_max*512
    if box[3] > h_max*512+1023:
        new_box[3] = 1023
    else:
        new_box[3] = box[3] - h_max*512
    name = filename_last[:-4]+'_CROPPED_'+str(n_max)+'.JPG'
    return new_box, name, [h_max, w_max], max_inters/area, n_max


def crop_and_save(df, dir):
    subdir = pathlib.Path(dir+'/CROPPED')
    subdir.mkdir(parents=True, exist_ok=True)
    with open('solace_crop.csv', 'w') as f:
        header = ['img_id', 'img_from_id', 'file', 'hw', 'box_id', 'box', 'label']
        writer = csv.writer(f)
        writer.writerow(header)
        for k, file in enumerate(df['file'].unique()):
            id_original = np.asarray(df[df['file'] == file]['box_id'])
            img_from_id = df[df['file'] == file]['img_id'].values[0]
            set_id = set(id_original)
            images, names, hw, boxes, boxes_id, labels, n_list, rejected_list, names_rejected = crop_image(file, dir, df)
            for n, img_cropped in enumerate(images):
                print(names[n], img_cropped.shape)
                io.imsave('./CROPPED/' + names[n], img_cropped)
                for m in range(len(boxes[n])):
                    data = [k*100 + n_list[n], img_from_id, names[n], hw[n], boxes_id[n][m], boxes[n][m], labels[n][m]]
                    writer.writerow(data)
            #visualize_detections(images[20], boxes[20], labels[20], scores=np.ones(len(labels[20])))
            set_id_crop = set(itertools.chain(*boxes_id))
            set_diff = set_id - set_id_crop
            for id in set_diff:
                box = df[df['box_id']==id]['box'].apply(lambda x: x.replace('[', '').replace(']', '')).values[0]
                box = np.asarray([float(el) for el in box.split(' ') if len(el) > 0], dtype=np.int32)
                #print(box, box[2]-box[0], box[3]-box[1], id)
                new_box, name, hw, area_ratio, n = check_rejected_box(box, file, dir)
                #print(id, area_ratio, n, new_box)
                label = df[df['box_id']==id]['label'].values[0]
                data = [k*100 + n, img_from_id, name, hw, id, new_box, label]
                writer.writerow(data)
                #print(new_box, name, label)
            for n, rejected in enumerate(rejected_list):
                if not os.path.isfile('./CROPPED/' + names_rejected[n]):
                    print(names_rejected[n], rejected.shape)
                    io.imsave('./CROPPED/' + names_rejected[n], rejected)


def write_train_test_id(df, test_rate, filename):
    indices = df['img_id'].unique()
    rand = np.random.permutation(indices)
    train_len = int(len(indices)*(1-test_rate))
    train_indices = rand[:train_len]
    test_indices = rand[train_len:]
    data = {
        "train": train_indices,
        "test": test_indices,
    }
    with open(filename, "wb") as resultFile:
        pickle.dump(data, resultFile)


def read_train_test_id(filename):
    with open(filename, "rb") as resultFile:
        data = pickle.load(resultFile)

    return data['train'], data['test']


def prepare_image(image):
    #image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0)#, ratio


def test_image(image, model, num_classes, boxes_orig=None, labels_orig=None):
    img = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(img, training=False)
    detections = DecodePredictions(confidence_threshold=0.5, num_classes=num_classes)(img, predictions)
    inference_model = tf.keras.Model(inputs=img, outputs=detections)
    image = tf.cast(image, dtype=tf.float32)
    input_image = prepare_image(image)
    #input_image = tf.keras.applications.resnet.preprocess_input(input_image)
    #input_image = tf.expand_dims(input_image, axis=0)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    print(num_detections)
    class_names = [
        class_list[label_list.index(int(x))] for x in detections.nmsed_classes[0][:num_detections]
    ]
    if labels_orig is not None:
        cls_orig = [class_list[label_list.index(int(x))] for x in labels_orig]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections],#/ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
        boxes_orig = boxes_orig,
        cls_orig = cls_orig
    )

"""url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")
tf.keras.utils.get_file(filename, url)


with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")

quit()"""
dir = os.getcwdb().decode('utf-8')
dir2 = os.path.join(dir, 'IMAGENES_ANALIZADAS')
#files, boxes, labels = load_images(dir2)
#quit()
"""size = []
ws = []
hs = []
for n in range(len(boxes)):
    for m in range(len(boxes[n])):
        w = abs(boxes[n][m][2]-boxes[n][m][0])
        h = abs(boxes[n][m][3]-boxes[n][m][1])
        size.append(w*h)
        ws.append(w)
        hs.append(h)
size = np.array(size)
print(np.max(hs), np.max(ws))
quit()"""
#dataset, label_encoder = data_to_dict(files, boxes, labels)
#data_to_csv(files, boxes, labels)
df = pd.read_csv('solace.csv')
#print(len(df_crop['img_id'].unique()))
#crop_and_save(df, dir)
#quit()
df_crop = pd.read_csv('solace_crop.csv')
#write_train_test_id(df, 0.1, 'train_test_id.pkl')
num_classes, class_dict = export_classes(df)
#print(num_classes)
#quit()
class_list = list(class_dict.keys())
label_list = list(class_dict.values())
#print(class_list)
#quit()
train_id, test_id = read_train_test_id('train_test_id.pkl')
#print(test_id, train_id)
#quit()
crop_train_id = []
crop_test_id = []
for id in train_id:
    crop_train_id.append(df_crop[df_crop['img_from_id']== id]['img_id'].unique())
for id in test_id:
    crop_test_id.append(df_crop[df_crop['img_from_id']== id]['img_id'].unique())
crop_train_id = list(itertools.chain(*crop_train_id))
crop_test_id = list(itertools.chain(*crop_test_id))

train_dataset = csv_to_dataset(df_crop, crop_train_id)
test_dataset = csv_to_dataset(df_crop, crop_test_id).shuffle(1000)
test_imgs = []
test_boxes = []
test_labels = []
count_classes = []
for data in test_dataset.take(10):
    #count_classes.append(data[2].numpy())
    #visualize_detections(data[0], data[1], data[2], np.ones(len(data[2])))
    test_imgs.append(data[0])
    test_boxes.append(data[1])
    test_labels.append(data[2])
#count_total = list(itertools.chain(*count_classes))
#print(pd.Series(count_total).value_counts())
#print(class_dict)
#quit()
#plt.show()
#quit()
#colors_classes = []
#for n in range(len(classes)):
#    colors_classes.append('#%06X' % random.randint(0, 0xFFFFFF))

""""(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)"""
"""for data in train_dataset.take(5):
    print(data)
    #anchor = AnchorBox()
    #anchor.get_anchors(data['image'].shape[0], data['image'].shape[1])
quit()"""
model_dir = "retinanet/resnet"
batch_size = 2

label_encoder = LabelEncoder()


learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
#optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
model.compile(loss=loss_fn, optimizer=optimizer)

#scores = np.ones(len(labels[0]))
#visualize_detections(images[0], boxes[0], labels[0], scores, classes, colors_classes, figsize=(12,12))
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights"),#filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
]
autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_solace, num_parallel_calls=autotune)
#train_dataset = train_dataset.shuffle(8 * batch_size)
#train_dataset = train_dataset.padded_batch(
#    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
#)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

test_dataset = test_dataset.map(preprocess_solace, num_parallel_calls=autotune)
test_dataset = test_dataset.batch(1)
test_dataset = test_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors())
test_dataset = test_dataset.prefetch(autotune)

"""val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)"""

# Uncomment the following lines, when training on full dataset
# train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // batch_size

# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch
#dataset_to_numpy = train_dataset.as_numpy_iterator()

epochs = 5
sample = 10

weights_dir = model_dir
latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset
"""model.fit(
    train_dataset,
    validation_data=test_dataset,#val_dataset.take(5),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
    batch_size=batch_size,
    #steps_per_epoch=sample//batch_size,
    shuffle = True
)"""
#model.save('ocean_detection_h5', overwrite=True, include_optimizer=True, save_format='tf')
# Change this to `model_dir` when not using the downloaded weights


"""val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
int2str = dataset_info.features["objects"]["label"].int2str
#print(dataset_info.features["objects"]["label"].names)
#quit()

img = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(img, training=False)
detections = DecodePredictions(confidence_threshold=0.5, num_classes=num_classes)(img, predictions)
inference_model = tf.keras.Model(inputs=img, outputs=detections)

for sample in val_dataset.shuffle(5).take(1):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )

quit()"""
for n, test_img in enumerate(test_imgs):
    test_image(test_img, model, num_classes, boxes_orig=None, labels_orig=test_labels[n])
plt.show()