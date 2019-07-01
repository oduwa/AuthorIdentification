from __future__ import division
import pickle
import io
import os, sys
import csv
import re
import nltk
from string import punctuation
import nltk.tokenize
import numpy as np

def removePunctuation(text):
    '''
    Removes punctuation, changes to lower case and strips leading and trailing
    spaces.

    Args:
        text (str): Input string.

    Returns:
        (str): The cleaned up string.
    '''
    text.strip()
    return ''.join(c for c in text.decode('ascii', 'ignore') if c not in punctuation or c in ['#','\n','?','!','\''])

def preprocess_text(txt):
    '''
    Applies preprocessing operations to a given text and returns preprocessed text.
    '''
    # Lower case
    txt = txt.lower()

    # remove urls
    txt = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', txt)

    txt = removePunctuation(txt)

    return txt

def serialize(obj, ser_filename):
    '''
    Saves a python object to disk.

    If the object being dealt with is a list, the contents of thenew list get
    added to the existing serialized list. Otherwise, the new object ovewrites
    the old one.

    Args:
        obj: Object to save.
        ser_filename: Filename to save object with on disk.
        isSerializingList: Boolean denoting whether object to be saved is a list
                            or not.
    '''
    f = open(ser_filename, 'wb')
    pickle.dump(obj, f)
    f.close()

def unserialize(ser_filename):
    '''
    Loads a python object from disk.

    Returns:
        The python object at the specified path or None if none is found.
    '''
    if(not os.path.isfile(ser_filename)):
        return None
    else:
        f = open(ser_filename, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj

def is_serialized_object_in_path(path):
    return not(unserialize(path) == None)

def bigram_tokenize(txt):
    bigram_list = []
    tokenized_msg = nltk.tokenize.word_tokenize(txt)
    for i in range(len(tokenized_msg)-1):
        bigram = tokenized_msg[i] + " " + tokenized_msg[i+1]
        bigram_list.append(bigram)
    return bigram_list

def misclassified_indices(truth,observed):
    misclassified_indices = np.where(np.asarray(truth) != np.asarray(observed))
    return misclassified_indices[0]
