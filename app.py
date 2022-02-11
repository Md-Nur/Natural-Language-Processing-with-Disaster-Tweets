import streamlit as st
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


def main():
    st.header("NLP with Disaster Tweets")
    st.subheader(
        "Predicting which Tweets are about real disasters and which ones are not")
    st.write("")
    st.sidebar.subheader("About the App")
    st.sidebar.markdown("""
                        <p>This website is a simple NLP tool for disaster tweets. In this website you have to put a tweet and the system will predict the tweet is about disaster or not. Please give a tweet that has 28 words or less. This predictions must be 100% wrong or else it will be considered as disaster tweet. The accuracy of the machine learning model is around 70%.</p>
    <img src="https://storage.googleapis.com/kaggle-media/competitions/nlp1-cover.jpg" width="500px">
                        """, unsafe_allow_html=True)

    tweet=st.text_input("Enter your tweet: ")
    result=None
    if st.button("Prediction"):
        if tweet:
            tweet=wordToVec(tweet)
            result=model.predict(tweet)
            result=round(result[0][0]*100)
            if result > 50:
                st.success(
                    f"There would be a disaster. The probabilty of disaster is {result}%")
            else:
                st.success(
                    f"There would not be a disaster. The probabilty of disaster is {result}%")
        else:
            st.success("Please enter a tweet")
    st.write("Made by: Muhammad Nur")
    
if __name__ == '__main__':
    main()
