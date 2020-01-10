import tensorflow as tf



def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature((), tf.string),
        "one_hot_label" : tf.io.FixedLenFeature((), tf.string),
        "label": tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    image = tf.reshape(image, [-1])
    one_hot_label = tf.io.parse_tensor(example['one_hot_label'], out_type = tf.int64)  # already a byte string
    return image, tf.argmax(one_hot_label)



def create_dataset(record_dir, batch_size = 32):

    '''
        record_dir: directory for the tfrecords
    '''
    # read from tfrecs
    train_filenames = tf.data.Dataset.list_files(record_dir + "train*.tfrecords")
    train_records = tf.data.TFRecordDataset(train_filenames)
    train_dataset = train_records.map(read_tfrecord).repeat().shuffle(10000).batch(batch_size)

    valid_filenames = tf.data.Dataset.list_files(record_dir + "valid*.tfrecords")
    valid_records = tf.data.TFRecordDataset(valid_filenames)
    valid_dataset = valid_records.map(read_tfrecord)
    
    
    return train_dataset, valid_dataset


def read_metadata_tfrecord(example):
    features = {
        "age": tf.io.FixedLenFeature((), tf.float32),
        "site": tf.io.FixedLenFeature((), tf.string),
        "sex": tf.io.FixedLenFeature((), tf.string),
        "one_hot_label" : tf.io.FixedLenFeature((), tf.string),
        "label": tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(example, features)
    age = example['age'] / 100.0
    age = tf.expand_dims(age, 0)
    site = tf.io.parse_tensor(example['site'], out_type = tf.float32)
    sex = tf.io.parse_tensor(example['sex'], out_type = tf.float32)
    dx = example['label']
    one_hot_label = tf.io.parse_tensor(example['one_hot_label'], out_type = tf.int64)  # already a byte string
    
    features = tf.concat([age, site, sex], 0)
    
    return features, tf.argmax(one_hot_label)

def create_meta_dataset(record_dir, batch_size = 32):
    
    meta_train_filenames = tf.data.Dataset.list_files(record_dir + "metadata_train.tfrecords")
    meta_train_records = tf.data.TFRecordDataset(meta_train_filenames)
    meta_train_dataset = meta_train_records.map(read_metadata_tfrecord).repeat().batch(128)
    
    meta_valid_filenames = tf.data.Dataset.list_files(record_dir + "metadata_valid.tfrecords")
    meta_valid_records = tf.data.TFRecordDataset(meta_valid_filenames)
    meta_valid_dataset = meta_valid_records.map(read_metadata_tfrecord).batch(128)
    
    
    return meta_train_dataset, meta_valid_dataset


    