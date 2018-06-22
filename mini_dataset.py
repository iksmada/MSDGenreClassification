#  use data/MSD_genre/msd_genre_dataset.txt as features
# !/usr/bin/python3
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn import under_sampling, combine


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
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize="smaller",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    start_time = time()

    parser = argparse.ArgumentParser(description='Million Song Dataset Genre Classification')
    parser.add_argument('-i', '--input', type=str, help='Input data path',
                        default="data/MSD_genre/msd_genre_dataset.csv")
    parser.add_argument('-s', '--size', help='Train size in % relative to test set',
                        default=0.8)

    args = vars(parser.parse_args())
    print(args)
    INPUT = args["input"]
    TRAIN_SIZE = args['size']

    # Load data
    X =[]
    fieldnames = []
    with open(INPUT, newline='') as csvfile:
        csvreader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
        fieldnames = next(csvreader)
        print(fieldnames)
        for row in csvreader:
            X.append(row)
    # Convert to numpy array
    X = np.array(X)
    # Remove song info and split classes and data
    y, _, X = np.split(X, [1, 4], axis=1)

    # encode labels
    le = LabelEncoder()
    le.fit(y)
    y_transformed = le.transform(y)
    # string to float
    X = X.astype(float)
    # divide dataset
    combination = combine.SMOTETomek(n_jobs=-1)
    X, y_transformed = combination.fit_sample(X, y_transformed)
    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, train_size=TRAIN_SIZE, stratify=y_transformed)

    # tribute to our biggest forest
    amazon = RandomForestClassifier(max_features="sqrt")

    # Grid Search number of trees
    # Range of `n_estimators` values to explore.
    n_features = X.shape[1]
    n_estim = list(range(10, min(2*n_features, 100), 2))

    cv_scores = []

    for i in n_estim:
        amazon.set_params(n_estimators=i)
        # 5x2 cross-validation
        kfold = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)
        scores = cross_val_score(amazon, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
        cv_scores.append(scores.mean())

    optimal_n_estim = n_estim[cv_scores.index(max(cv_scores))]
    print("The optimal number of estimators is %d with %0.1f%%" % (optimal_n_estim, max(cv_scores)*100))

    plt.plot(n_estim, cv_scores)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Train Accuracy')
    plt.show()

    amazon.set_params(n_estimators=optimal_n_estim, n_jobs=-1)
    amazon.fit(X_train, y_train)
    print(amazon.feature_importances_)

    y_pred = amazon.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cmat = confusion_matrix(y_test, y_pred)
    cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2, suppress=True)
    #print(le.classes_)
    #print(cmat)
    acc_per_class = cmat.diagonal() / cmat.sum(axis=1)
    print("Accuracy on test set of %d samples: %f" % (len(y_test), accuracy_score(y_test, y_pred)))
    print("Normalized Accuracy on test set: %f" % (np.mean(acc_per_class)))
    print("F1 Score on test set: %f" % (f1_score(y_test, y_pred, average="macro")))
    plt.figure()
    plot_confusion_matrix(cmat, classes=le.classes_, title='Random Forest Confusion matrix')
    plt.show()

    print("--- %s seconds ---" % (time() - start_time))
