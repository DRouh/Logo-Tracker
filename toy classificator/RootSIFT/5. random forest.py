from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
import glob
import pickle
import os
import sys

if __name__ == "__main__":
    try:
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


        # Extremely Random forest
        pipeline = Pipeline([
            ('clf', ExtraTreesClassifier(criterion='entropy', verbose=0))
        ])

        parameters = {
            'clf__n_estimators': (100, 200, 400, 800, 1600, 10000),
            'clf__max_depth': (25, 50, 100, 200, 250, 300),
            'clf__min_samples_split': (1, 2, 3,),
            'clf__min_samples_leaf': (1, 2, 3,)
        }

        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0, scoring='f1')
        grid_search.fit(X_train, y_train)

        print 'Best score: %0.3f' % grid_search.best_score_
        print 'Best parameters set:'
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print '\t%s: %r' % (param_name, best_parameters[param_name])

        predictions = grid_search.best_estimator_.predict(X_test)

        print classification_report(y_test, predictions)
        print 'Precision: ', precision_score(y_test, predictions)
        print 'Recall: ', recall_score(y_test, predictions)
        print 'Accuracy: ', accuracy_score(y_test, predictions)
        with open('ERF_searchgrid.pkl', 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)
        with open('ERF_best.pkl', 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)

        # Random forest
        pipeline2 = Pipeline([
            ('clf', RandomForestClassifier(criterion='entropy', verbose=0))
        ])
        grid_search2 = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0, scoring='f1')
        grid_search2.fit(X_train, y_train)

        print 'Best score: %0.3f' % grid_search2.best_score_
        print 'Best parameters set:'
        best_parameters = grid_search2.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print '\t%s: %r' % (param_name, best_parameters[param_name])

        predictions = grid_search2.best_estimator_.predict(X_test)

        print classification_report(y_test, predictions)
        print 'Precision: ', precision_score(y_test, predictions)
        print 'Recall: ', recall_score(y_test, predictions)
        print 'Accuracy: ', accuracy_score(y_test, predictions)
        with open('RF_searchgrid.pkl', 'wb') as f:
            pickle.dump(grid_search2.best_estimator_, f)
        with open('RF_best.pkl', 'wb') as f:
            pickle.dump(grid_search2.best_estimator_, f)

        #os.system("shutdown /s")
    except:
        os.system("shutdown /s")