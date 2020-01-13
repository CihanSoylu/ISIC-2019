import tensorflow as tf
import argparse
import create_model
import create_datasets
import numpy as np
import accuracy
import os
import itertools


from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score



desc_string = "Set possible values of the hyperparameters and train on all combinations."


def run_training(log_dir,
                 checkpoint_path,
                 train_dataset,
                 valid_dataset,
                 epochs,
                 batch_size,
                 model_name,
                 regularization,
                 unit_size):


    callbacks = [tf.keras.callbacks.TensorBoard(log_dir = log_dir),
                tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  verbose = 1,
                                                  save_best_only = True)]

    steps_per_epoch = 23331 // batch_size
    class_weights = {0: 2.8008624502432555,
                     1: 0.9837281553398058,
                     2: 3.8114655431838704,
                     3: 14.608419838523645,
                     4: 4.826791158536586,
                     5: 52.99372384937238,
                     6: 50.06126482213438,
                     7: 20.16799363057325}


    model = create_model.get_model(regularization, unit_size)

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

    epochs = 250
    base_models = ['densenet201']

    batch_sizes = [128]

    regularizations = [1e-15]

    #unit_sizes = [[256, 128], [512, 256], [1024, 512], [2048, 1024], [4096, 2048], [256, 128, 64], [512, 256, 128], [1024, 512, 256], [2048, 1024, 512]]
    unit_sizes = [[2048, 1024]]


    hyper_parameters = list(itertools.product(base_models, batch_sizes, unit_sizes, regularizations))

    for base_model, batch_size, unit_size, reg in hyper_parameters:

        tfrecords_dir = 'data/tfrecords/' + base_model + '-valid-2000/'

        if reg:
            regularization = tf.keras.regularizers.l2(l=reg)
        else:
            regularization = None

        log_dir =  'logs5/' + base_model + '-batch-{}'.format(batch_size) + '-regularization-{}'.format(reg) + '-unitsize-' + '-'.join([str(x) for x in unit_size])
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        checkpoint_path = log_dir + '/cp.ckpt'


        train_dataset, valid_dataset = create_datasets.create_dataset(tfrecords_dir,
                                                                      batch_size = batch_size)

        trained_model = run_training(log_dir,
                                     checkpoint_path,
                                     train_dataset,
                                     valid_dataset.batch(batch_size),
                                     epochs,
                                     batch_size,
                                     base_model,
                                     regularization,
                                     unit_size)

        accuracy.test_accuracy(trained_model, valid_dataset)

        trained_model.save_weights(log_dir + '/last_model.h5')

        best_model = trained_model.load_weights(checkpoint_path)
        accuracy.test_accuracy(trained_model, valid_dataset)
