import tensorflow as tf
import os
import argparse
import glob
import pandas as pd
import numpy as np
from progress.bar import Bar


desc_string = "Create test tfrecords"
parser = argparse.ArgumentParser(description=desc_string)
parser.add_argument(
    '-bm', '--base_model', type=str, required=True, default = 'densenet201',
    help="Pretrained Base Model"
)
args = parser.parse_args()


DATA_DIR = os.getcwd() + '/data'


test_images = glob.glob("data/ISIC_2019_Test_Input/*.jpg")


def get_base_model(base_model_name):
    if base_model_name == 'xception':
        input_shape = (299, 299, 3)
        base_model = tf.keras.applications.Xception(include_top = False,
                                                    weights = 'imagenet',
                                                    input_shape = input_shape,
                                                    pooling = 'avg')
        preprocess = tf.keras.applications.xception.preprocess_input
    elif base_model_name == 'densenet201':
        input_shape = (224, 224, 3)
        base_model = tf.keras.applications.DenseNet201(include_top = False,
                                                       weights = 'imagenet',
                                                       input_shape = input_shape,
                                                       pooling = 'avg')
        preprocess = tf.keras.applications.densenet.preprocess_input
    elif base_model_name == 'inception_resnet_v2':
        input_shape = (299, 299, 3)
        base_model = tf.keras.applications.InceptionResNetV2(include_top = False,
                                                             weights = 'imagenet',
                                                             input_shape = input_shape,
                                                             pooling = 'avg')
        preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input

    return base_model, preprocess, input_shape

base_model, preprocess, input_shape = get_base_model(args.base_model)

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_example_record(img_id, bottleneck_features):
    feature = {'id': _bytes_feature(img_id),
               'image': _bytes_feature(bottleneck_features)}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()

def create_tfrecords(img_paths):

    record_dir = DATA_DIR + '/tfrecords/test'
    if not os.path.isdir(record_dir):
        os.mkdir(record_dir)

    #Write the validation data
    #record_count = 0
    #shard = 0

    bar = Bar("Creating tfrecords", max=len(img_paths))

    filename = record_dir + '/' + '{}.tfrecords'.format(args.base_model)  # address to save the TFRecords file
        #    # open the TFRecords file
    writer = tf.io.TFRecordWriter(filename)

    for img_path in img_paths:
        bar.next()


        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        #make some data augmentation/preprocessing here

        x = preprocess(x)
        bottleneck_features = base_model.predict(x)

        bottleneck_features = tf.io.serialize_tensor(bottleneck_features)
        img_id = tf.compat.as_bytes(img_path[26:38])


        # Serialize to string and write on the file
        writer.write(create_example_record(img_id, bottleneck_features))

    writer.close()
    bar.finish()

if __name__ == '__main__':

    create_tfrecords(test_images)
