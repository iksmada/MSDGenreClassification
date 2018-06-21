#  use data/MSD_genre/msd_genre_dataset.txt as features
# !/usr/bin/python3
#  -*- coding: UTF-8 -*-

import argparse
import sys
import os
import csv
from time import time
import numpy as np

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    start_time = time()

    parser = argparse.ArgumentParser(description='Million Song Dataset Genre Classification')
    parser.add_argument('-i', '--input', type=str, help='Input data path',
                        default="data/MSD_genre/msd_genre_dataset.csv")
    parser.add_argument('-s', '-size', help='Train size in % relative to test set',
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE)



    print("--- %s seconds ---" % (time() - start_time))
