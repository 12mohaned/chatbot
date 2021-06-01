import pandas as pd
from nltk import NaiveBayesClassifier, classify
from sklearn.model_selection import train_test_split
import os
import pickle

fileObj = open('suicide_data.csv', 'rb')
exampleObj = pickle.load(fileObj)
fileObj.close()
train_data_set = exampleObj[:8000]
test_data_set = exampleObj[8000:]
classifier = NaiveBayesClassifier.train(train_data_set)
pickled_model = open('naive_model.sav', 'wb')
pickle.dump(classifier,pickled_model)
pickled_model.close()
