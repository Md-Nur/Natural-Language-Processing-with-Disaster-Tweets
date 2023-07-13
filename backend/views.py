from django.http import HttpResponse
from django.shortcuts import render
from tensorflow import keras
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = keras.models.load_model('model.h5')


def wordToVec(tweet):
    vocSize = 5000
    nltk.download('stopwords')
    ps = PorterStemmer()
    corpus = []

    review = re.sub('[^a-zA-Z]', ' ', str(tweet))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)
    for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    oneHotTrain = [one_hot(i, vocSize) for i in corpus]
    senLen = 28
    embededTrain = pad_sequences(oneHotTrain, maxlen=senLen, padding='pre')
    return embededTrain

def home(request):
    tweet = " "
    data={
        # 'bgColor':"bg-lime-500",
        # 'bgColor':"bg-red-500",
        'bgColor':" ",
        "result":"Your Result will show here"
    }
    try:
        name = request.GET['name']
        location = request.GET['location']
        tweet = request.GET['tweet']
    except:
        name = " "
        location = " "
        tweet = " "
    if tweet != " ":
        tweet=wordToVec(tweet)
        result=model.predict(tweet)
        result=round(result[0][0]*100)
        if result > 50:
            data['bgColor']="bg-red-500"
            data['result']=f"{name}, there would be a disaster in {location}. The probabilty of disaster is {result}%"
        else:
            data['bgColor']="bg-lime-500"
            data['result']=f"{name}, there would not be a disaster in {location}. The probabilty of disaster is {result}%"
        return render(request, "index.html", data)
    else:
        data['result'] = "Please enter a tweet to show the result"
        data['bgColor'] = " "
        return render(request, "index.html", data)
