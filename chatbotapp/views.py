from django.shortcuts import render
from json import dumps
import aiml
import time

def init():
    kernel = aiml.Kernel()
    kernel.learn("std-startup.xml")
    kernel.respond("load aiml b")
    return kernel


def chat(request):
        if(rule_based("الا")):
            print("found in rulebased")
        else:
            print("syntheseing a reply")
        return render(request,"chatbotapp/channel.html")

def rule_based(message):
        chatbot = init()
        input_text = input(">Human: ")
        response = chatbot.respond(message)
        return len(response) > 0
