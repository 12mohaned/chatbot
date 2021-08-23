from django.shortcuts import render
from json import dumps
import aiml
import time
import pathlib
import pickle
import requests

from django.http import JsonResponse

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

    def is_stop_word(word):
        if word in stop_words():
            return True
        return False

def init():
    kernel = aiml.Kernel()
    kernel.learn("std-startup.xml")
    kernel.respond("load aiml b")
    return kernel


async def chat(request):
        curr_message = ""
        if request.method == 'POST':
            message=request.POST.get('task')
            lang_detect = await language_detection(message)
            if lang_detect:
                if(pattern_found(message)):
                    chatbot = init()
                    response = chatbot.respond(message)
                    curr_message = response
                else:
                    print("syntheseing a reply")
            else:
                print("The chatbot only support arabic language")
        return render(request,"chatbotapp/channel.html",{"message":JsonResponse(curr_message,safe = False)})

def pattern_found(message):
        chatbot = init()
        response = chatbot.respond(message)
        return len(response) > 0

async def language_detection(message):
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

def de_serialize_model():
    pickled_model = open('chatbotapp/emotion_model.sav', 'rb')
    naive_bayes_classifier = pickle.load(pickled_model)
    pickled_model.close()
    return naive_bayes_classifier
