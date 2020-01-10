import tensorflow as tf
import argparse
import create_model
import create_datasets
import numpy as np
import accuracy
import os

from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score



desc_string = "Run training on a set of hyperparameters provided by commandline arguments."
parser = argparse.ArgumentParser(description=desc_string)
parser.add_argument(
    '-md', '--model_directory', type=str, required=False, default = 'logs',
    help="Directory to save the model and logs"
) # Takes None if not specified
parser.add_argument(
    '-bm', '--base_model', type=str, required=True, default = 'densenet201',
    help="Pretrained Base Model"
)
parser.add_argument(
    '-ep', '--epochs', type=int, required=True, default = 10,
    help="Number of epochs"
)
parser.add_argument(
    '-bs', '--batch_size', type=int, required=True, default = 32,
    help="Batch size"
)
parser.add_argument(
    '-rg', '--regularization', type=float, required = False, default = 0,
    help="Regularization option"
)
args = parser.parse_args()


def run_training(log_dir,
                 checkpoint_path,
                 train_dataset,
                 valid_dataset,
                 epochs,
                 batch_size,
                 model_name,
                 regularization):


    callbacks = [tf.keras.callbacks.TensorBoard(log_dir = log_dir),
                tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  verbose = 1,
                                                  save_best_only = False)]

    steps_per_epoch = 23331 // batch_size
    class_weights = {3: 3.6536321989528795,
                     2: 0.9530129737111642,
                     4: 1.2068201469952442,
                     5: 13.229265402843602,
                     0: 0.7001191371958866,
                     1: 0.24593612334801762,
                     7: 5.038583032490974,
                     6: 12.51737668161435}


    model = create_model.get_model(regularization, [2048, 1024])

    model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              verbose=1,
              callbacks = callbacks,
              validation_data=valid_dataset,
              validation_steps=None,
              class_weight = class_weights)


    return model


if __name__ == '__main__':

    tfrecords_dir = 'data/tfrecords/' + args.base_model + '-valid-2000/'

    if args.regularization:
        regularization = tf.keras.regularizers.l2(l=args.regularization)
    else:
        regularization = None

    log_dir = args.model_directory + '/' + args.base_model + '-batch-{}'.format(args.batch_size) + '-regularization-{}'.format(args.regularization)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    checkpoint_path = log_dir + '/cp.ckpt'


    train_dataset, valid_dataset = create_datasets.create_dataset(tfrecords_dir,
                                                                  batch_size = args.batch_size)

    trained_model = run_training(log_dir,
                                 checkpoint_path,
                                 train_dataset,
                                 valid_dataset.batch(args.batch_size),
                                 args.epochs,
                                 args.batch_size,
                                 args.base_model,
                                 regularization)

    accuracy.test_accuracy(trained_model, valid_dataset)

    best_model = trained_model.load_weights(checkpoint_path)
    accuracy.test_accuracy(trained_model, valid_dataset)
