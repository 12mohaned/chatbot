import nltk
import os
import pickle
import re as regx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk import word_tokenize

stop_words = stopwords.words('Arabic')

def preprocess(text):
    text = regx.sub("[إأآا]", "ا", text)
    text = regx.sub("ى", "ي", text)
    text = regx.sub("ؤ", "ء", text)
    text = regx.sub("ئ", "ء", text)
    text = regx.sub("ة", "ه", text)
    text = regx.sub("گ", "ك", text)

    ret_text = text
    text_tokenized = word_tokenize(text)
    for word in text_tokenized:
        if not is_stop_word(word):
            ret_text+=word
    return ret_text

def tokenize(text):
    return nltk.word_tokenize(text)

def is_stop_word(word):
    if word in stop_words:
        return True
    return False

data = pd.read_excel('data/40000-Egyptian-tweets.xlsx', engine='openpyxl')
df = pd.DataFrame(data = data)
df.isnull().sum()
df['label'].unique()
df["label"] = df["label"].astype("string")
df['label'].replace({'negative':0,'positive':1}, inplace = True)
df.drop(df[df['label'] != 0].index & df[df['label'] != 1].index, inplace = True)
df["label"] = df["label"].astype("int")
df['review'] = df['review'].apply(preprocess)
vectorizer = CountVectorizer(analyzer = 'word',tokenizer = tokenize,ngram_range=(1, 1), )
predictor = df['review']
target = df['label']
X_train, X_test, Y_train, Y_test = train_test_split(predictor, target, test_size =.3, random_state=100)
naive_pipe = make_pipeline(TfidfVectorizer(),MultinomialNB())
naive_pipe.fit(X_train,Y_train)
pickled_model = open('emotion_model.sav', 'wb')
pickle.dump(naive_pipe,pickled_model)
pickled_model.close()
