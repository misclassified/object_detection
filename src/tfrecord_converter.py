from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import sys

sys.path.append("../../models/research")

from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from collections import namedtuple, OrderedDict


class TfRecordConverter(object):

    def __init__(self, images_path, images_xml_df, label_map_path, output_path):
        """Class to convert images and annotations to XML

        Arguments:
            images_path = str, path where the images to convert are stored
            images_xml_df = pandas df with reference to files and annotations
                created with the utilis to converto_xml_to_csv function
            label_map_path = str, path where a protobuf label map file is stored
            output_path = str, path where tfr records are stored
        """
        self.images_path = images_path
        self.xml_df = images_xml_df
        self.label_map_path = label_map_path
        self.output_path = output_path

    def create_tfrecord(self):
        """Main function to create TF Records"""

        # Inizialize the TFR Record Writer
        writer = tf.python_io.TFRecordWriter(self.output_path)
        path = os.path.join(self.images_path)

        # Create label map dictionary
        label_map = label_map_util.load_labelmap(self.label_map_path)

        categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)

        category_index = label_map_util.create_category_index(categories)

        label_map = {}
        for k, v in category_index.items():
            label_map[v.get("name")] = v.get("id")

        # Create groups
        grouped = self._split()

        # Create TFR Record
        for group in grouped:
            tf_example = self._create_tf_example(group, label_map)
            writer.write(tf_example.SerializeToString())

        writer.close()
        print('TFRecords created in {}'.format(self.output_path))


    def _split(self):
        """Convenience function that input a pandas
        dataframe of xml annotations related to an image
        and split into several smaller dataframe for each image.

        To create the xml use the convert_xml_to_csv in the
        utils package.

        Arguments:
            df = pandas dataframe
        """

        data = namedtuple("data", ["filename", "object"])
        gb = self.xml_df.groupby("filename")

        groups = [
            data(filename, gb.get_group(x))
            for filename, x in zip(gb.groups.keys(), gb.groups)
        ]

        return groups

    def _create_tf_example(self, group, label_map):
        """Convert an images and respective annotation
        to TFR Record (a binary file formata)

        Arguments:
            group = pandas data frame
            label_map = dict
        """

        # Serialize Image
        with tf.gfile.GFile(os.path.join(
            self.images_path, "{}".format(group.filename)), "rb") as fid:
            encoded_jpg = fid.read()

        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode("utf8")
        image_format = b"jpg"

        # check if the image format is matching with your images.
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        # Serialize XML annotation
        for index, row in group.object.iterrows():
            xmins.append(row["xmin"] / width)
            xmaxs.append(row["xmax"] / width)
            ymins.append(row["ymin"] / height)
            ymaxs.append(row["ymax"] / height)
            classes_text.append(row["class"].encode("utf8"))
            class_index = label_map.get(row["class"])
            assert (
                class_index is not None
            ), "class label: `{}` not found in label_map: {}".format(
                row["class"], label_map
            )
            classes.append(class_index)

        # Create Tf Record
        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": dataset_util.int64_feature(height),
                    "image/width": dataset_util.int64_feature(width),
                    "image/filename": dataset_util.bytes_feature(filename),
                    "image/source_id": dataset_util.bytes_feature(filename),
                    "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                    "image/format": dataset_util.bytes_feature(image_format),
                    "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                    "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                    "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                    "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                    "image/object/class/text": dataset_util.bytes_list_feature(
                        classes_text
                    ),
                    "image/object/class/label": dataset_util.int64_list_feature(classes),
                }
            )
        )
        return tf_example
