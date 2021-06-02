from django.shortcuts import render
from json import dumps
import aiml
import time
import pathlib
import pickle
import requests

def stopwords():
    return stopwords.words('Arabic')

def preprocessing_pipeline():
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
        if word in stop_words():
            return True
        return False

def init():
    kernel = aiml.Kernel()
    kernel.learn("std-startup.xml")
    kernel.respond("load aiml b")
    return kernel


def chat(request):
        curr_message = ""
        if request.method == 'POST':
            message=request.POST.get('task')
            if(rule_based(message)):
                chatbot = init()
                response = chatbot.respond(message)
                curr_message = response
            else:
                print("syntheseing a reply")

        return render(request,"chatbotapp/channel.html",{"data":curr_message})

def rule_based(message):
        chatbot = init()
        response = chatbot.respond(message)
        return len(response) > 0
def de_serialize_model():
    pickled_model = open('chatbotapp/emotion_model.sav', 'rb')
    naive_bayes_classifier = pickle.load(pickled_model)
    pickled_model.close()
    return naive_bayes_classifier
