#  use data/MSD_genre/msd_genre_dataset.txt as features
# !/usr/bin/python3 -W ignore::DeprecationWarning
#  -*- coding: UTF-8 -*-

import argparse
from time import time

import numpy as np
from sklearn.cluster import KMeans


def kmeans_reclass(X, y):
    n_classes = len(np.unique(y))
    model = KMeans(n_clusters=n_classes, init='k-means++', max_iter=500, n_init=10, n_jobs=-1)
    y_pred = model.fit_predict(X)

    # separate clusters
    clusters = dict()
    orig = 0
    for pred in y_pred:
        if pred not in clusters:
            clusters[pred] = []
        clusters[pred].append(y[orig])
        orig = orig + 1

    counts = np.zeros((n_classes, n_classes), dtype=int)
    for pred, cluster in clusters.items():
        for orig in range(n_classes):
            # create matrix columns are originals classes and row are clusters
            counts[pred][orig] = cluster.count(orig)

    conversor = dict()
    #get biggest to smallest cluster
    for pred in np.argsort(-np.sum(counts, axis=1)):
        orig = np.argmax(counts[pred, :])
        conversor[pred] = orig
        counts[:, orig] = np.zeros(counts[:, orig].shape)

    y_reclass = []
    for pred in y_pred:
        y_reclass.append(conversor[pred])

    return np.array(y_reclass)


if __name__ == '__main__':
    start_time = time()

    parser = argparse.ArgumentParser(description='Million Song Dataset Genre Classification')
    parser.add_argument('-i', '--input', type=str, help='Input data path',
                        #default="data/MSD_genre/msd_genre_dataset.csv")
                        default="extractedLetters/all_letters.csv")
    parser.add_argument('-s', '--size', type=float, help='Train size in % relative to test set',
                        default=0.8)
    parser.add_argument('-t', '--tree', type=int, help='Number of tree')
    parser.add_argument('-n', '--features', type=int, help='Number of features from the beginning')

    print("--- %s seconds ---" % (time() - start_time))
