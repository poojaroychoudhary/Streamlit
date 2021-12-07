import streamlit as st

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#from wordcloud import WordCloud,STOPWORDS
from wordcloud import WordCloud, STOPWORDS
#import create_wordcloud
import nltk

from PIL import Image

st.title('INNOMATICS RESEARCH LABS')
st.title('INTERNSHIP 2021')
st.sidebar.title('SENTIMENT ANALYSIS OF AIRLINES')
st.sidebar.markdown("WE CAN ANALYSE PASSENGERS REVIEW FROM THIS APPLICATION")

st.subheader('NLP PROJECT')
st.title('AIRLINE SENTIMENT ANALYSIS')

image = Image.open('airline.jpeg')

st.image(image, caption='OVER THE SKY')
st.title('INTRODUCTION')
st.subheader('This application is all about tweet sentiment analysis of airlines. We can analyse reviews of the passengers using the streamlit app')
st.title('OBJECTIVE OF PROJECT')
st.subheader('Analyse how travellers in February 2015 expressed their feelings on twitter')
st.title('Load the dataset')


df = pd.read_csv('data\Tweets.csv')
st.dataframe(df)

st.header("Data Preview")
st.subheader('Checking The Head & Tail Of The Dataset')
preview = st.radio("Choose Head/Tail?", ("Top", "Bottom"))
if(preview == "Top"):
            st.write(df.head())
if(preview == "Bottom"):
            st.write(df.tail())



def load_data(df):
    df1 = df.loc[:, ['airline_sentiment', 'airline', 'text']]
    return df1

df_new = load_data(df)

st.header("DISPLAY THE WHOLE DATASET")
if(st.checkbox("Show complete Dataset")):
            st.write(df_new)

st.subheader('Created the new dataframe df_new with the column names as airline_sentiment, airline and text')

# Show shape
if(st.checkbox("Display the shape")):
            st.write(df_new.shape)
            dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
            if(dim == "Rows"):
                st.write("Number of Rows", df_new.shape[0])
            if(dim == "Columns"):
                st.write("Number of Columns", df_new.shape[1])


st.subheader('Columns of df_new')
st.write(df_new.columns)

st.title("VISUALISATION")
st.subheader('Tweet Sentiment Count')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write(sns.countplot(x="airline_sentiment",data=df_new))

st.pyplot()

st.title("Airline Count")

st.write(sns.countplot(x="airline",data=df_new))

st.pyplot()

st.subheader("Airline Count")
airline = st.radio("Choose an Airline?", ("US Airways", "United", "American", "Southwest", "Delta", "Virgin America"))
temp_df = df.loc[df['airline']==airline, :]
st.write(sns.countplot(x='airline_sentiment', order=['neutral', 'positive', 'negative'], data=temp_df))
st.pyplot()

select = st.sidebar.selectbox('Visualisation Of Tweets',['Barplot','Pie Chart'],key = 1)

sentiment = df['airline_sentiment'].value_counts()
sentiment = pd.DataFrame({'Sentiment':sentiment.index,'Tweets':sentiment.values})
st.markdown("### Sentiment count")

if select == "Barplot":
    fig = px.bar(sentiment, x='Sentiment', y='Tweets',color = 'Tweets',height = 500)
    st.plotly_chart(fig)
else:
    fig = px.pie(sentiment,values='Tweets',names='Sentiment')
    st.plotly_chart(fig)


st.subheader('It is clearly depicted by the pie chart that there are 62.7% of negative sentiments, 16.1% of positive sentiments and 21.2% are neutral sentiments')


st.sidebar.markdown("Airline tweets by sentiment")
choice = st.sidebar.multiselect("Airlines",('US Airways','United','American','Southwest','Delta','Virgin America'),key='0')
if len(choice)>0:
    air_data = df[df.airline.isin(choice)]
    fig1 = px.histogram(air_data,x='airline',y='airline_sentiment',histfunc='count',color='airline_sentiment',labels='airline_sentiment')
    st.plotly_chart(fig1)

st.sidebar.subheader('Tweets Analyser')

tweets = st.sidebar.radio('Sentiment Type',('positive','negative','neutral'))

st.write (df.query('airline_sentiment == @tweets')[['text']].sample(1).iat[0,0])
st.write (df.query('airline_sentiment == @tweets')[['text']].sample(1).iat[0,0])
st.write (df.query('airline_sentiment == @tweets')[['text']].sample(1).iat[0,0])
st.write (df.query('airline_sentiment == @tweets')[['text']].sample(1).iat[0,0])
st.write (df.query('airline_sentiment == @tweets')[['text']].sample(1).iat[0,0])
st.write (df.query('airline_sentiment == @tweets')[['text']].sample(1).iat[0,0])
st.write (df.query('airline_sentiment == @tweets')[['text']].sample(1).iat[0,0])


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords') 

from pickle import dump, load
#
classifier_loc = "pickle/logit_model.pkl"
encoder_loc = "pickle/countvectorizer.pkl"
image_loc = "twitter_img.jpg"


# WordCloud
# def load_wordcloud(df, kind):
#     temp_df = df.loc[df['airline_sentiment']==kind, :]
#     words = ' '.join(temp_df['text'])
#     cleaned_word = " ".join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
#     wc = WordCloud(stopwords=STOPWORDS, background_color='black', width=1600, height=800).generate(cleaned_word)
#     return wc


# def main():
#     nltk.download('stopwords')
#     wc = load_wordcloud(df_new, "positive")
#     wc.to_file("img/pos.png")


#     wc = load_wordcloud(df_new, "negative")
#     wc.to_file("img/neg.png")

# if(__name__=="__main__"):
#     main()


##########################################################################################################################################################

st.header("Word Cloud of Positive and Negative Tweets")

preview = st.radio("Choose Positive Tweets/Negative Tweets", ("Positive", "Negative"))
if(preview == "Positive"):
            st.image("img/pos.png", use_column_width = True)
if(preview == "Negative"):
            st.image("img/neg.png", use_column_width = True)







##########################################################################################################################################################



def preprocess(tweet):
     # Removing special characters and digits
     letters_only = re.sub("[^a-zA-Z]", " ",tweet)

     # change sentence to lower case
     letters_only = letters_only.lower()

     # tokenize into words
     words = letters_only.split()

     # remove stop words
     words = [w for w in words if not w in stopwords.words("english")]

     # Stemming
     stemmer = PorterStemmer()
     words = [stemmer.stem(word) for word in words]

     clean_sentence = " ".join(words)

     return clean_sentence



def predict(tweet):

     # Loading pretrained CountVectorizer from pickle file
     vectorizer = load(open('pickle/countvectorizer.pkl', 'rb'))

     # Loading pretrained logistic classifier from pickle file
     classifier = load(open('pickle/logit_model.pkl', 'rb'))

     # Preprocessing the tweet
     clean_tweet = preprocess(tweet)

     # Converting text to numerical vector
     clean_tweet_encoded = vectorizer.transform([clean_tweet])

     # Converting sparse matrix to dense matrix
     tweet_input = clean_tweet_encoded.toarray()

     # Prediction
     prediction = classifier.predict(tweet_input)

     return prediction



def main():

     st.image("twitter_img.jpg", use_column_width = True)

     tweet = st.text_input('Enter your tweet')

     prediction = predict(tweet)

     if(tweet):
         st.subheader("Prediction:")
         if(prediction == 0):
             st.write("Negative Tweet :cry:")
         else:
             st.write("Positive Tweet :sunglasses:")


if(__name__ == '__main__'):
     main()












