import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score


def test_accuracy(model, test_dataset):

    features, labels = next(iter(test_dataset.batch(2000)))
    labels = labels.numpy()
    predictions = model.predict(features)
    predictions = np.argmax(predictions, axis = 1)
    test_accuracy = 100*np.sum(predictions == labels)/len(predictions)
    print('\nTest accuracy: %.2f%%' % test_accuracy)

    print(confusion_matrix(labels, np.array(predictions)))
    print(classification_report(labels, np.array(predictions)))
    balanced_accuracy = balanced_accuracy_score(labels, np.array(predictions))
    print('\nBalanced accuracy: {:.3f}'.format(balanced_accuracy) )
