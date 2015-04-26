import glob
import pickle

from sklearn.metrics import *
from sklearn.metrics import classification_report


if __name__ == "__main__":
    # Load the images
    train_instance_filenames = []
    train_instance_targets = []
    for f in glob.glob('train_hist/*.hist'):
        target = 1 if 'cocacola' in f else 0
        train_instance_filenames.append(f)
        train_instance_targets.append(target)
    print

    test_instance_filenames = []
    test_instance_targets = []
    for f in glob.glob('test_hist/*.hist'):
        target = 1 if 'cocacola' in f else 0
        test_instance_filenames.append(f)
        test_instance_targets.append(target)
    print

    surf_features = []
    counter = 1

    X_train = []
    y_train = train_instance_targets
    for f in train_instance_filenames:
        X_train.append(pickle.load(open(f, "rb")))
        counter += 1

    counter = 1
    X_test = []
    y_test = test_instance_targets
    for f in test_instance_filenames:
        X_test.append(pickle.load(open(f, "rb")))
        counter += 1

    clf = pickle.load(open("GB_best.pkl", "rb"))
    predictions = clf.predict(X_test)

    print classification_report(y_test, predictions)
    print 'Precision: ', precision_score(y_test, predictions)
    print 'Recall: ', recall_score(y_test, predictions)
    print 'Accuracy: ', accuracy_score(y_test, predictions)