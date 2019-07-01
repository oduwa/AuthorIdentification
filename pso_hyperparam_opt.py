# -*- coding: utf-8 -*-
r"""Hyperparameter optimization using evolutionary algorithms.

This script implements hyperparameter optimization using Particle Swarm
Optimization (PSO).
"""
from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import log_loss
from pyswarm import pso
import Helper
import numpy as np
import argparse
import wine_model

parser = argparse.ArgumentParser()
parser.add_argument(
  '--vratio',
  type=float,
  default=0.3,
  help='fraction of test set to use for validation'
)
args = vars(parser.parse_args())
VALIDATION_RATIO = args['vratio']

trainX = None
trainY = None
testX = None
testY = None
classes = None

def get_dataset_partitions():
    '''
    Loads WineModel dataset and splits it into its training/test/validation
    sets, and returns the training and validation sets for use in hyperparameter
    optimization.

    @Returns tuple (train_texts, train_targets, validation_texts, validation_targets)
    train_texts and validation_texts are lists containing text description strings
    and train_targets and validation_targets are lists containing corresponding
    target classes.
    '''
    global classes
    # Load test set
    wm = wine_model2.WineModel()
    dataset = wm.get_dataset()

    # Partition dataset
    texts = [rvw['text'] for rvw in dataset]
    targets = [rvw['taster_name'] for rvw in dataset]
    classes = list(set(targets))

    train_texts = texts[:int(len(dataset)*wm.get_train_ratio())]
    train_targets = targets[:int(len(dataset)*wm.get_train_ratio())]
    test_texts = texts[int(len(dataset)*wm.get_train_ratio()):]
    test_targets = targets[int(len(dataset)*wm.get_train_ratio()):]
    validation_texts = test_texts[int(len(test_texts)*VALIDATION_RATIO):]
    validation_targets = test_targets[int(len(test_targets)*VALIDATION_RATIO):]

    return train_texts, train_targets, validation_texts, validation_targets

def build_featureset(train_texts, train_targets, validation_texts, validation_targets):
    '''
    Constructs feature vector representation of dataset and returns it.

    @Return tuple (trainX, trainY, testX, testY)
    trainX and testX are ;ists of feature vectors of text descriptions and
    trainY and testY are lists of corresponding target classes.
    '''
    global trainX, trainY, testX, testY

    # Create bag of words vectorizer to use scipy.sparse matrices
    # and avoid wasting memory storing the many non-zero entries in the
    # bag.
    count_vect = CountVectorizer()
    X_counts_train = count_vect.fit_transform(train_texts)

    # Apply tf-idf
    tfidf_transformer = TfidfTransformer()
    trainX = tfidf_transformer.fit_transform(X_counts_train)
    trainY = train_targets

    # Repeat for test set
    X_counts_test = count_vect.transform(validation_texts)
    testX = tfidf_transformer.transform(X_counts_test)
    testY = validation_targets

    return trainX, trainY, testX, testY

def loss(x):
    '''
    Objective function that is minimised by PSO.

    @param x list A list representing a particle. Its ith position contains the
    ith hyperparameter being optimised

    @Returns float value of objective function (which in this case is
    log loss)
    '''
    hyperparam_a = x[0]
    hyperparam_b = x[1]

    # Train
    classifier = LogisticRegression(C=hyperparam_a, intercept_scaling=hyperparam_b).fit(trainX, trainY)

    # Apply
    preds = classifier.predict_proba(testX)

    # Compute loss
    loss = log_loss(testY, preds, labels=classes)

    return loss

if __name__ == '__main__':
    # Load dataset
    train_texts, train_targets, validation_texts, validation_targets = get_dataset_partitions()

    # Convert dataset to feature representation
    build_featureset(train_texts, train_targets, validation_texts, validation_targets)

    # Dictate upper and lower bounds of a particle
    lower_bound = [0.01, 0.1]
    upper_bound = [1.2, 2]

    # Run PSO
    xopt, fopt = pso(loss, lower_bound, upper_bound, swarmsize=5, maxiter=5, omega=0.75, phip=0.75, phig=0.75, debug=True)
    print("OPTIMAL HYPERPARAMETER SETUP: {}\nMINIMUM LOSS OBTAINED: {}".format(xopt,fopt))
