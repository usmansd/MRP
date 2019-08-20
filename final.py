from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd
import csv
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import ClassifierChain
import re
import sys
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys
import os
import xml.etree.ElementTree as ET
output = 'output/en/'
loaded_model = pickle.load(open('finalized_model_humanOrBot.sav', 'rb'))
vectorizer = pickle.load(open('finalized_vectorizer_humanOrBot', 'rb'))
gender_vectorizer = pickle.load(open('finalized_vectorizer_gender', 'rb'))
gender_model = pickle.load(open('finalized_model_gender_LR.sav', 'rb'))

stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight',
                   'nine','ten','may','also','across','among','beside','however',
                   'yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

# STEMMER
stemmer = SnowballStemmer("english")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

def most_common(lst):
    lst = lst.tolist()
    return max(set(lst), key=lst.count)

def writeToFile(id, bothuman, gender):
    print(id,bothuman,gender)
    strr = '<author id=\"'+str(id)+'\"\nlang=\"en\"\ntype=\"'
    if bothuman == 0:
        strr = strr + 'bot\"\ngender=\"bot\"\n/>'
        f = open(output+id+'.xml','w+')
        f.write(strr)
        f.close()
        return
    else:
        strr = strr + 'human\"\ngender=\"'
    if gender == 0:
        strr = strr + 'female\"\n/>'
        f = open(output+id+'.xml','w+')
        f.write(strr)
        f.close()
        return
    if gender == 1:
        strr = strr + 'male\"\n/>'
        f = open(output+id+'.xml','w+')
        f.write(strr)
        f.close()
        return

def process(id, ts):
    tweets = []
    for t in ts:
        t = t.lower()
        t = clean_text(t)
        t = cleanHtml(t)
        t = cleanPunc(t)
        t = keepAlpha(t)
        t = removeStopWords(t)
        t = stemming(t)
        tweets.append(t)
    
    x_test = vectorizer.transform(tweets)
    predictions = loaded_model.predict(x_test)
    humanBot = most_common(predictions)
    if humanBot == 1:
        x_test = gender_vectorizer.transform(tweets)
        genders = gender_model.predict(x_test)
        gender = most_common(genders)
        writeToFile(id, humanBot, gender)
    else:
        writeToFile(id, humanBot, 2)

if __name__ == '__main__':
    #input_ = sys.argv[1]
    #output = sys.argv[2]
    input_ = 'input/'
    input_ = input_+'en/'
    files = os.listdir(input_)


    for file in files:
        tweets = []
        tree = ET.parse(input_+file)  
        root = tree.getroot()
        for child in root:
            for sub in child:
                tweets.append(sub.text)
        id = file.split('.')[0]
        process(id,tweets)
        print(id)
