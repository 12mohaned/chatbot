from flask import Flask, render_template, request
import aiml
import time
import pathlib
import pickle
import requests
from nltk.corpus import stopwords
from nltk import word_tokenize
import re as regx

app = Flask(__name__)
app.static_folder = 'static'

sentiment = []

def stop_words():
    return stopwords.words('Arabic')

def preprocess(text):
    text = regx.sub("[إأآا]", "ا", text)
    text = regx.sub("ى", "ي", text)
    text = regx.sub("ؤ", "ء", text)
    text = regx.sub("ئ", "ء", text)
    text = regx.sub("ة", "ه", text)
    text = regx.sub("گ", "ك", text)
    text = text
    ret_text = ""
    text_tokenized = word_tokenize(text)
    for word in text_tokenized:
        if not is_stop_word(word):
            ret_text+=word
            ret_text+=" "
    text = ret_text
    return text

def is_stop_word(word):
    if word in stop_words():
        return True
    return False

def init():
    kernel = aiml.Kernel()
    kernel.learn("std-startup.xml")
    kernel.respond("load aiml b")
    return kernel

def pattern_found(message):
        chatbot = init()
        response = chatbot.respond(message)
        return len(response) > 0
    
def de_serialize_model():
    pickled_model = open('emotion_model.sav', 'rb')
    naive_bayes_classifier = pickle.load(pickled_model)
    pickled_model.close()
    return naive_bayes_classifier

def language_detection(message):
    url = "https://api.meaningcloud.com/lang-4.0/identification"
    payload={
        'key': '4ebfba4a6f37ce3df9a5491cec15ce1a',
        'txt': message
    }
    response = requests.post(url, data=payload)
    output = response.json()
    language_list = output['language_list']
    language_nam  = language_list[0]['name']
    language_relevance = language_list[0]['relevance']
    return language_nam == "Arabic" and language_relevance >=70

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response = "عفو ,لم يتم التعرف علي الجمله هل يمكن كتابه شي اخر"
    model = de_serialize_model()
    userText = preprocess(userText)
    sentiment.append(userText)    
    if language_detection(userText):
        if pattern_found(userText):
            chatbot = init()
            response = chatbot.respond(userText)
    else:
        response = "عفو الدكتور يتحدث اللغه العربيه فقط"
    return str(response)

if __name__ == "__main__":
    app.run() 