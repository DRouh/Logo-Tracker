from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
import glob
import pickle

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
    print 'Reading image:{0}. {1}/{2}'.format(f, counter, len(train_instance_targets))
    X_train.append(pickle.load(open(f, "rb")))
    counter += 1

counter = 1
X_test = []
y_test = test_instance_targets
for f in test_instance_filenames:
    print 'Reading image:{0}. {1}/{2}'.format(f, counter, len(test_instance_targets))
    X_test.append(pickle.load(open(f, "rb")))
    counter += 1

# Train a logistic regression classifier on the feature vectors and targets,
# and assess its precision, recall, and accuracy.
print "Logistic regression"
clf = LogisticRegression(C=0.001, penalty='l2')
clf.fit_transform(X_train, y_train)
predictions = clf.predict(X_test)
print classification_report(y_test, predictions)
print 'Precision: ', precision_score(y_test, predictions)
print 'Recall: ', recall_score(y_test, predictions)
print 'Accuracy: ', accuracy_score(y_test, predictions)
with open('LogisticRegressionHard.pkl', 'wb') as f:
    pickle.dump(clf, f)