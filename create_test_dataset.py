import tensorflow as tf

def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature((), tf.string),
        "id": tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    image = tf.reshape(image, [-1])
    img_id = example['id']
    return image, img_id



def create_dataset(record_dir, base_model):

    '''
        record_dir: directory for the tfrecords
    '''
    # read from tfrecs
    filenames = tf.data.Dataset.list_files(record_dir + "{}.tfrecords".format(base_model))
    records = tf.data.TFRecordDataset(filenames)
    dataset = records.map(read_tfrecord)

    return dataset