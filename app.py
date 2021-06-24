from flask import Flask, render_template, request
import aiml
import time
import pathlib
import pickle
import requests
from nltk.corpus import stopwords
from nltk import word_tokenize
import re as regx
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from six.moves import cPickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from keras import backend as K
from nltk.stem import ISRIStemmer
from keras.utils import np_utils
from strsimpy.cosine import Cosine

app = Flask(__name__)
app.static_folder = 'static'
Keywords = {
  "اكتئاب": "الاكتئاب هو اضطراب مزاجي يسبب شعورًا دائمًا بالحزن وفقدان الاهتمام. ويسمى أيضًا اضطراب اكتئابي رئيسي أو اكتئاب سريري، وهو يؤثر على شعورك وتفكيرك وسلوكك ويمكن أن يؤدي إلى مجموعة متنوعة من المشاكل العاطفية والجسدية. قد تواجهك صعوبة في القيام بالأنشطة اليومية العادية، وأحيانًا قد تشعر كما لو أن الحياة لا تستحق العيش.",
  "انتحار": "إذا كنت تشعر بضغط شديد بسبب الأفكار التي تراودك عن عدم الرغبة في الحياة، أو تخطر لك أفكار مُلِحة عن محاولة الانتحار، فاطلب المساعدة الآن",
  "قلق":  "أحيانًا تكون المعاناة من القلق جزءًا طبيعيًّا من الحياة. ومع ذلك، فإنَّ الأشخاص الذين يعانون من اضطرابات القلق غالبًا ما يكون لديهم مخاوف وخوف مفرط ومستمر من المواقف اليومية. وفي كثير من الأحيان، تتضمن اضطرابات القلق نوبات متكررة من المشاعر المفاجئة للقلق الشديد والخوف أو الرعب، وتصل إلى ذروتها في غضون دقائق (نوبات الهلع)."
   "تشمَل العلامات والأعراض الشائعة للقلق ما يلي:" +
   "الشعور بالعصبيَّة أو القلق أو التوتُّر" + 
   "الشعور بالخطر الوَشيك أو الذُّعر أو التَّشاؤم" + 
   "الإصابة بزيادة مُعدَّل ضربات القلب" +
   "زيادة مُعدَّل التنفُّس (فرط التهوية)" +
   "التركيز على الصُّعوبات أو التفكير في أيِّ أمرٍ بِخلاف القلق الحالي.",
   "التوتر" : "بعض أبرز أسباب التوتر عند معظم الأفراد"
    "الأسباب الخارجية للتوتر: نتيجة التغييرات في الحياة العملية أو المدرسة أو الصعوبات المادية أو مشاكل الأطفال والعائلة." +
    "الأسباب الداخلية للتوتر: يحدث نتيجة التشاؤم المستمر والحديث السلبي أو التأثر بنظرة المجتمع بك." +
    " توتر مرحلة المراهقة: مثل طلاق الوالدين، فقدان أحد المقربين والتغيرات الجسمانية.",
    "panic attack":  "أحيانًا تكون المعاناة من القلق جزءًا طبيعيًّا من الحياة. ومع ذلك، فإنَّ الأشخاص الذين يعانون من اضطرابات القلق غالبًا ما يكون لديهم مخاوف وخوف مفرط ومستمر من المواقف اليومية. وفي كثير من الأحيان، تتضمن اضطرابات القلق نوبات متكررة من المشاعر المفاجئة للقلق الشديد والخوف أو الرعب، وتصل إلى ذروتها في غضون دقائق (نوبات الهلع)."
   "تشمَل العلامات والأعراض الشائعة للقلق ما يلي:" +
   "الشعور بالعصبيَّة أو القلق أو التوتُّر" + 
   "الشعور بالخطر الوَشيك أو الذُّعر أو التَّشاؤم" + 
   "الإصابة بزيادة مُعدَّل ضربات القلب" +
   "زيادة مُعدَّل التنفُّس (فرط التهوية)",
   "علاج" :" ننصح بضروره بالذهاب الي طبيب نفسي متخصص هذه بعض المواقع التي ممكن ان تساعد" + 
   "١- موقع شيزلونج:https://shezlong.com/"+
   "٢- موقع علاج نفسي:https://ilajnafsy.wordpress.com/tag/علاج-نفسي-مجاني/ "+
   "٣- موقع Cups7: https://www.7cups.com/",
   "مساعده" :" ننصح بضروره بالذهاب الي طبيب نفسي متخصص هذه بعض المواقع التي ممكن ان تساعد" + 
   "١- موقع شيزلونج:https://shezlong.com/"+
   "٢- موقع علاج نفسي:https://ilajnafsy.wordpress.com/tag/علاج-نفسي-مجاني/ "+
   "٣- موقع Cups7: https://www.7cups.com/",
   "تشخيص" :" ننصح بضروره بالذهاب الي طبيب نفسي متخصص هذه بعض المواقع التي ممكن ان تساعد" + 
   "١- موقع شيزلونج:https://shezlong.com/"+
   "٢- موقع علاج نفسي:https://ilajnafsy.wordpress.com/tag/علاج-نفسي-مجاني/ "+
   "٣- موقع Cups7: https://www.7cups.com/"
}
sentiment = []

def sentence_similarity(textinput):
    cosine = Cosine(2)
    for word in word_tokenize(textinput):
     for keyword in Keywords:
            if cosine.similarity_profiles(cosine.get_profile(keyword),cosine.get_profile(word)) > 0.80:
                return Keywords[keyword]
    return ""

def stop_words():
    return stopwords.words('Arabic')

def preprocess(text):
    text = regx.sub("[إأآا]", "ا", text)
    text = regx.sub("ى", "ي", text)
    text = regx.sub("ؤ", "ء", text)
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

def de_serialize_lstm(input):
    K.set_learning_phase(False)
    MAX_LEN = 125
    data = pd.read_csv("data/arabic-empathetic-conversations.csv")
    tokenizer = cPickle.load(open("models/lstm-autoencoder-tokenizer.pickle", "rb"))
    stemmer = ISRIStemmer()
    model = load_model("models/lstm-encoder.h5")
    encode = K.function([model.input], [model.layers[1].output])
    Questions = tokenizer.texts_to_sequences(data.context)
    Questions = pad_sequences(Questions, padding='post', truncating='post', maxlen=MAX_LEN)
    Questions = np.squeeze(np.array(encode([Questions])))
    question = stemmer.stem(input)
    question = tokenizer.texts_to_sequences([question])
    question = pad_sequences(question, padding='post', truncating='post', maxlen=MAX_LEN)
    question = np.squeeze(encode([question]))

    rank = cosine_similarity(question.reshape(1, -1), Questions)
    top = np.argsort(rank, axis=-1).T[-5:].tolist()
    for item in top:
        return data['response'].iloc[item].values[0]
    
def de_serialize_emotion_model():
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
    response = "عفو الدكتور يتحدث العربيه فقط"
    userText = preprocess(userText)
    sentiment.append(userText)    
    
    if language_detection(userText):
        if pattern_found(userText):
            chatbot = init()
            response = chatbot.respond(userText)
        else:
            if len(sentence_similarity(userText)) > 0:
                response = sentence_similarity(userText)
            else:
                response = de_serialize_lstm(userText)
    return str(response)

if __name__ == "__main__":
    app.run() 