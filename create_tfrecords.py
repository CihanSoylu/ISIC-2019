import tensorflow as tf
import os
import argparse
import glob
import pandas as pd
import numpy as np
from progress.bar import Bar
from sklearn.model_selection import train_test_split


desc_string = "Create tfrecords from image files"
parser = argparse.ArgumentParser(description=desc_string)
parser.add_argument(
    '-vs', '--validation_size', type=int, required=True, default = '2000',
    help="Validation dataset size"
) # Takes None if not specified
parser.add_argument(
    '-bm', '--base_model', type=str, required=True, default = 'densenet201',
    help="Pretrained Base Model"
)
args = parser.parse_args()


DATA_DIR = os.getcwd() + '/data'


ground_truth_df = pd.read_csv(DATA_DIR + '/ISIC_2019_Training_GroundTruth.csv')

CLASS_NAMES = ground_truth_df.columns.values[1:]
NCLASSES = len(CLASS_NAMES)


def one_hot(labels):
    return np.array([int(x) for x in labels[CLASS_NAMES].values])

def dx(labels):
    return CLASS_NAMES[labels['one_hot'] == 1][0]

ground_truth_df['one_hot'] = ground_truth_df.apply(one_hot, axis = 1)
ground_truth_df['dx'] = ground_truth_df.apply(dx, axis = 1)
ground_truth_df = ground_truth_df.sample(frac = 1)

train_df, valid_df, train_idx, valid_idx = train_test_split(ground_truth_df,
                                                          ground_truth_df['dx'],
                                                          stratify = ground_truth_df['dx'],
                                                          test_size = args.validation_size)     

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

def create_example_record(dx, one_hot_label, bottleneck_features):
    feature = {'label': _bytes_feature(dx),
               'one_hot_label': _bytes_feature(one_hot_label),
               'image': _bytes_feature(bottleneck_features)}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()

def create_tfrecords(df, name = None):

    record_dir = DATA_DIR + '/tfrecords/{}-valid-{}'.format(args.base_model, args.validation_size)
    if not os.path.isdir(record_dir):
        os.mkdir(record_dir)

    #Write the validation data
    #record_count = 0
    #shard = 0

    bar = Bar("Creating %s tfrecords" % name, max=len(df))

    filename = record_dir + '/' + name + '.tfrecords'  # address to save the TFRecords file
        #    # open the TFRecords file
    writer = tf.io.TFRecordWriter(filename)

    for row in df.itertuples():
        bar.next()

        img_path = DATA_DIR + '/ISIC_2019_Training_Input/' + row.image + '.jpg'
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        #make some data augmentation/preprocessing here

        x = preprocess(x)
        bottleneck_features = base_model.predict(x)

        bottleneck_features = tf.io.serialize_tensor(bottleneck_features)
        dx = tf.compat.as_bytes(row.dx)
        one_hot_label = tf.io.serialize_tensor(row.one_hot)


        # Serialize to string and write on the file
        writer.write(create_example_record(dx, one_hot_label, bottleneck_features))

    writer.close()
    bar.finish()

if __name__ == '__main__':

    create_tfrecords(valid_df, name = 'valid')
    create_tfrecords(train_df, name = 'train')
