#  use data/MSD_genre/msd_genre_dataset.txt as features
# !/usr/bin/python3 -W ignore::DeprecationWarning
#  -*- coding: UTF-8 -*-

import argparse
import sys
import os
import csv
from time import time
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, scorer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from imblearn import under_sampling, combine


from reclass import kmeans_reclass

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, horizontalalignment="right",)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(np.round(cm[i, j]*100)),
                 horizontalalignment="center",
                 fontsize="smaller",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def f1_encoder(y, y_pred, **kwargs):
    lb = LabelBinarizer()
    y_binarized = lb.fit_transform(y)
    y_pred_bin = lb.transform(y_pred)
    return f1_score(y_binarized, y_pred_bin, average='macro')


def print_distribution(y, classes):
    values, counts = np.unique(y, return_counts=True)
    for clazz, count in zip(values, counts):
        print("%s: %d,\t" % (classes[clazz], count), end='')
    print('')


if __name__ == '__main__':
    start_time = time()

    parser = argparse.ArgumentParser(description='Million Song Dataset Genre Classification')
    parser.add_argument('-i', '--input', type=str, help='Input data path',
                        default="data/MSD_genre/msd_genre_dataset.csv")
                        #default="extractedLetters/all_letters.csv")
    parser.add_argument('-s', '--size', type=float, help='Train size in % relative to test set',
                        default=0.8)
    parser.add_argument('-t', '--tree', type=int, help='Number of tree')
    parser.add_argument('-n', '--features', type=int, help='Number of features from the beginning')
    parser.add_argument('--reclass', action='store_true', help='Use reclassification algorithm')

    args = vars(parser.parse_args())
    print(args)
    INPUT = args["input"]
    TRAIN_SIZE = args['size']
    TREE = args['tree']
    FEATURES = args['features']
    RECLASS = args['reclass']

    # Load data
    X = []
    y = []
    with open(INPUT, newline='') as csvfile:
        csvreader = csv.reader(filter(lambda row: row[0] != '#', csvfile))
        fieldnames = next(csvreader)
        print(fieldnames)
        for row in csvreader:
            y.append(row[0])
            # Remove song info and split classes and data
            row = np.nan_to_num(np.array(row[4:(FEATURES+4 if FEATURES else None)]).astype(float))
            X.append(row)
    # Convert to numpy array
    X = np.asarray(X)
    y = np.array(y)
    # Remove irrelevant features - track_id,artist_name,title,duration
    #X = np.delete(X, 5, 1)
    # One hot encode categorical variables - time_signature,key 2 e 3
    # implicit string to float
    #enc = OneHotEncoder(categorical_features=[2, 3], sparse=False)
    #X = enc.fit_transform(X)

    # encode labels
    le = LabelEncoder()
    y_transformed = le.fit_transform(y)
    # divide dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, train_size=TRAIN_SIZE, stratify=y_transformed)
    print("Test set has %d samples" % len(y_test))
    print("Train set has %d samples" % len(y_train))
    print_distribution(y_train, le.classes_)

    # Resample the train set
    resampler = combine.SMOTETomek(n_jobs=-1)
    X_train, y_train = resampler.fit_sample(X_train, y_train)
    print("Resampled train set has %d samples" % len(y_train))
    print_distribution(y_train, le.classes_)

    if RECLASS:
        y_train = kmeans_reclass(X_train, y_train)
        print("Reclassified train set has %d samples" % len(y_train))
        print_distribution(y_train, le.classes_)

    # tribute to our biggest forest
    amazon = RandomForestClassifier(max_features="sqrt")

    if TREE:
        optimal_n_estim = TREE
    else:
        # Grid Search number of trees
        # Range of `n_estimators` values to explore.
        n_features = X_train.shape[1]
        n_estim = list(range(max(int(n_features/5), 10), min(n_features, 200)+1, 5))

        cv_scores = []
        my_scorer = scorer.make_scorer(f1_encoder, greater_is_better=True)
        print("Testing trees:", end=" ")
        for i in n_estim:
            print(i, end=' ', flush=True)
            amazon.set_params(n_estimators=i)
            # 5x2 cross-validation
            kfold = RepeatedStratifiedKFold(n_repeats=1, n_splits=2)
            scores = cross_val_score(amazon, X_train, y_train, cv=kfold, scoring=my_scorer, n_jobs=-1)
            cv_scores.append(scores.mean())

        print("")
        optimal_n_estim = n_estim[cv_scores.index(max(cv_scores))]
        print("The optimal number of estimators is %d with %0.1f%%" % (optimal_n_estim, max(cv_scores)*100))

        plt.plot(n_estim, cv_scores)
        plt.xlabel('Number of Estimators')
        plt.ylabel('Train Accuracy')
        plt.show()

    amazon.set_params(n_estimators=optimal_n_estim, n_jobs=-1)
    amazon.fit(X_train, y_train)
    print(amazon.feature_importances_)

    np.set_printoptions(precision=2, suppress=True)
    y_pred = amazon.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cmat = confusion_matrix(y_test, y_pred)
    cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
    #print(le.classes_)
    #print(cmat)
    acc_per_class = cmat.diagonal() / cmat.sum(axis=1)
    print("Accuracy on test set of %d samples: %.2f" % (len(y_test), accuracy_score(y_test, y_pred)))
    print("Normalized Accuracy on test set: %.2f" % (np.mean(acc_per_class)))
    print("F1 Score on test set: %.2f" % (f1_score(y_test, y_pred, average="macro")))
    plt.figure()
    plot_confusion_matrix(cmat, classes=le.classes_, title='Random Forest Confusion matrix')
    plt.show()

    print("--- %s seconds ---" % (time() - start_time))
