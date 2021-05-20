from django.shortcuts import render
import aiml

def init():
    kernel = aiml.Kernel()
    kernel.learn("std-startup.xml")
    kernel.respond("load aiml b")
    return kernel

def chat(request):
    # while True:
    #     chatbot = init()
    #     input_text = input(">Human: ")
    #     response = chatbot.respond(input_text)
    #     print(">Bot: "+response)
    return render(request,"chatbotapp/channel.html")
# Create your views here.
