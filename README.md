# ISIC-2019

This repository consists of the code I used to create my submission file for [ISIC-2019 Challenge](https://challenge2019.isic-archive.com/).
Brief summary of the method can be found in `method.pdf`.

## Data

In order to run the code first you need to download the [data](https://challenge2019.isic-archive.com/) and place the files inside /data.
The data folder should include 'ISIC_2019_Training_GroundTruth.csv', 'ISIC_2019_Training_Input' and 'ISIC_2019_Test_Input'.

See Data_exploration.ipynb for more info on the data.

We will save the bottleneck features obtained from the pretrained model as tfrecords. For this run

`python create_tfrecords.py`

Default validation data size is 2000 and pretrained model base model is densenet201.
Use the commandline arguments -vs and -bm to change the validation size and the base model.
See the code for the available pretrained models.

This will create a folder /tfrecords under /data.

## Training

To train a model use with particular hyperparameters use training.py

`python training.py -md logs -ep 20 -bs 32 -bm densenet201`

The arguments for training.py are:

 - -md --model_directory: The directory to save the model checkpoints and tensorboard file. Default is /logs
 - -ep --epochs: Number of epochs. Default is 10.
 - -bs --batch_size: Batch size. Default is 32.
 - -bm --base_model: The pretrained model used for transfer learning. Default is densenet201.
 - -rg --regularization: The parameter for L2 regularization. Default is 0.

In order to run training with multiple hyperparameter configurations use hyperparameter_tuning.py

`python hyperparameter_tuning.py`

You can change the possible values of different hyperparameters in hyperparameter_tuning.py.

## Submission

In order to get the predictions for the test data, first create tfrecords from test images using test_tfrecords.py:

`python test_tfrecords.py`

In the notebook 'create-submission.ipynb', we load a trained model, create the predictions and submission.csv.
See the notebook for the details.
