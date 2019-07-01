# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
import nltk.tokenize
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import csv
import re,sys
import Helper
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import argparse
import random
import collections
from sklearn.metrics import precision_recall_fscore_support

parser = argparse.ArgumentParser()
parser.add_argument(
  '--eval',
  type=bool,
  default=False,
  help='flag controlling whether to run evaluations'
)
args = vars(parser.parse_args())
should_eval = args['eval']

TRAIN_RATIO = 0.8
DATA_PATH = "serial/dataset.ser"
MODEL_PATH = "serial/model.ser"
COUNT_VECTORIZER_PATH = "serial/count.ser"
TFIDF_VECTORIZER_PATH = "serial/tfidf.ser"
TRAIN_X_PATH = "serial/X_train.ser"
TRAIN_Y_PATH = "serial/Y_train.ser"
TEST_X_PATH = "serial/X_test.ser"
TEST_Y_PATH = "serial/Y_test.ser"

class WineModel(object):
    def __init__(self):
        # LOAD DATASET
        if Helper.is_serialized_object_in_path(DATA_PATH):
            dataset = Helper.unserialize(DATA_PATH)
        else:
            dataset = self.__load_data()

        # BUILD MODEL
        self.classifier, self.count_vectorizer, self.tfidf_vectorizer = self.__build_model(dataset)

    def __load_data(self):
        '''
        Load dataset from CSV file.

        Loads wine review data from winemag-data-130k-v2.csv

        @return List<Dict> List of Review objects modelled as dictionaries with
                            the keys as column_names from data.
        '''
        self.reviews = []

        # LOAD DATA FROM DATASET
        with open('winemag-data-130k-v2.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['taster_name']:
                    rvw = {}
                    rvw['text'] = Helper.preprocess_text(row['description']).decode('utf-8')
                    rvw['taster_name'] = row['taster_name'].decode('utf-8').replace(u'’', u"")
                    rvw['points'] = int(row['points'])
                    self.reviews.append(rvw)

        Helper.serialize(self.reviews, DATA_PATH)
        return self.reviews

    def __build_model(self, dataset):
        '''
        Train model (or load from disk if already previously trained).

        @param List<Dict> dataset List of review objects modelled as dictionary objects with
                            the keys "text", "taster_name" and "points"

        @return (classifier, count_vectorizer, tfidf_vectorizer)
        '''
        count_vect = None
        tfidf_transformer = None

        # Get texts and target classes
        texts = [rvw['text'] for rvw in dataset]
        targets = [rvw['taster_name'] for rvw in dataset]

        train_texts = texts[:int(len(dataset)*TRAIN_RATIO)]
        train_targets = targets[:int(len(dataset)*TRAIN_RATIO)]
        test_texts = texts[int(len(dataset)*TRAIN_RATIO):]
        test_targets = targets[int(len(dataset)*TRAIN_RATIO):]

        if not Helper.is_serialized_object_in_path(MODEL_PATH):
            print("BUILDING FEATURE VECTORS...")
            # Create bag of words vectorizer to use scipy.sparse matrices
            # and avoid wasting memory storing the many non-zero entries in the
            # bag.
            count_vect = CountVectorizer()
            X_counts_train = count_vect.fit_transform(train_texts)

            # Apply tf-idf
            tfidf_transformer = TfidfTransformer()
            X_tfidf_train = tfidf_transformer.fit_transform(X_counts_train)
            print X_tfidf_train.shape

            Helper.serialize(count_vect, COUNT_VECTORIZER_PATH)
            Helper.serialize(tfidf_transformer, TFIDF_VECTORIZER_PATH)

            # Train classifier
            print("TRAINING CLASSIFIER...")
            classifier = LogisticRegression(C=1.0, intercept_scaling=1.0).fit(X_tfidf_train, train_targets)

            # Evaluate classifier
            if should_eval:
                print("EVALUATING CLASSIFIER...")
                X_counts_test = count_vect.transform(test_texts)
                X_tfidf_test = tfidf_transformer.transform(X_counts_test)
                predictions = classifier.predict(X_tfidf_test)
                print("ACCURACY: {}".format(np.mean(predictions == test_targets)))

                global_p, global_r, global_f, _ = precision_recall_fscore_support(test_targets, predictions, average='weighted', labels=classifier.classes_)
                local_p, local_r, local_f, _ = precision_recall_fscore_support(test_targets, predictions, average=None, labels=classifier.classes_)
                print("CLASS                  F1")
                for i,cls in enumerate(classifier.classes_):
                    print("{} {}".format(cls.encode('utf-8'), local_f[i]))
                print("MEAN PRECISION: {}\nMEAN RECALL: {}\nMEAN F1: {}".format(global_p, global_r, global_f))

            Helper.serialize(classifier, MODEL_PATH)
            return classifier, count_vect, tfidf_transformer
        else:
            # Load trained classifier
            print("LOADING CLASSIFIER...")
            classifier = Helper.unserialize(MODEL_PATH)
            count_vect = Helper.unserialize(COUNT_VECTORIZER_PATH)
            tfidf_transformer = Helper.unserialize(TFIDF_VECTORIZER_PATH)

            # Evaluate classifier
            if should_eval:
                print("EVALUATING CLASSIFIER...")
                X_counts_test = count_vect.transform(test_texts)
                X_tfidf_test = tfidf_transformer.transform(X_counts_test)
                predictions = classifier.predict(X_tfidf_test)
                print("ACCURACY: {}".format(np.mean(predictions == test_targets)))

                global_p, global_r, global_f, _ = precision_recall_fscore_support(test_targets, predictions, average='weighted', labels=classifier.classes_)
                local_p, local_r, local_f, _ = precision_recall_fscore_support(test_targets, predictions, average=None, labels=classifier.classes_)
                print("CLASS                  F1")
                for i,cls in enumerate(classifier.classes_):
                    print("{} {}".format(cls.encode('utf-8'), local_f[i]))
                print("MEAN PRECISION: {}\nMEAN RECALL: {}\nMEAN F1: {}".format(global_p, global_r, global_f))

            return classifier, count_vect, tfidf_transformer

    def get_dataset(self):
        '''
        Fetches dataset used within model.

        @return List<Dict> List of Review objects modelled as dictionaries with
                            the keys as column_names from data.
        '''
        return self.__load_data()

    def get_train_ratio(self):
        '''
        Fetches train ratio used in building model.

        @return floar fraction of dataset used for training.
        '''
        return TRAIN_RATIO

    def predict_review_author(self, rvw):
        '''
        Predict the author of a given review from known authors.

        @param Dict rvw text.

        @return Dict result dictionary where each key is a possible class and the values are
                    corresponding class probability predictions.
        '''
        # CLASSIFY WITH MODEL
        X_counts = self.count_vectorizer.transform(rvw)
        X_tfidf = self.tfidf_vectorizer.transform(X_counts)
        class_probabilities = self.classifier.predict_proba(X_tfidf)
        prediction_class_label = np.argmax(class_probabilities)
        predicted_author = self.classifier.classes_[prediction_class_label]

        result = {"prediction":predicted_author, "confidence":class_probabilities[0][prediction_class_label]}
        return result


if __name__ == '__main__':
    # Playground test
    should_eval = True
    wm = WineModel()
    #print wm.predict_review_author(["Crushed thyme, alpine wildflower, beeswax and orchard-fruit aromas are front and center on this beautiful white. Proving just how well Pinot Bianco can do in Alto Adige, the creamy palate is loaded with finesse, delivering ripe yellow pear, creamy apple and citrus alongside tangy acidity. White-almond and stony mineral notes back up the finish."])
