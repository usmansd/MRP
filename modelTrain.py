from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import MultinomialNB
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
import pandas as pd

from sklearn.neural_network import MLPClassifier


#arrays for train
authors_en = []
tweets_en = []
isHuman_en = []
gender_en = []

#arrays for test
authors_en_test = []
tweets_en_test = []
isHuman_en_test = []
gender_en_test = []

# STOP WORDS
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight',
                   'nine','ten','may','also','across','among','beside','however',
                   'yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

# STEMMER
stemmer = SnowballStemmer("english")

########################---GLOBAL DECLARATIONS---##################################


########################---FUNCTIONS---############################################
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

if __name__=='__main__':
 
    print('Reading')
    trainSet = pd.read_csv('train.txt', sep=':::', engine='python')
    testSet = pd.read_csv('test.txt', sep=':::', engine='python')

    dataTrain = trainSet
    dataTest = testSet

    df = dataTrain.append(dataTest, ignore_index=True)
    filt = df['human'] == 1
    isHuman = df[filt]
    msk = np.random.rand(len(isHuman)) < 0.8
    train = isHuman[msk]
    test = isHuman[~msk]
    print('Vectorizing')
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(isHuman['tweet'].values.astype('U'))
    pickle.dump(vectorizer, open('finalized_vectorizer_humanOrBot_usman', 'wb'))
    print('Done train')
    input()
    print('Done train')

    print('Data transformation train')
    x_train = vectorizer.transform(isHuman['tweet'].values.astype('U'))
    y_train = isHuman.drop(labels = ['id','tweet','human'], axis=1)

    print('Data transformation test')
    x_test = vectorizer.transform(test['tweet'].values.astype('U'))
    y_test = test.drop(labels = ['id','tweet','human'], axis=1)

    
    classifier = MultinomialNB()
    # train
    classifier.fit(x_train, y_train.values.ravel())
    predictions = classifier.predict(x_test)
    print("Accuracy = ",accuracy_score(y_test,predictions))
    filename = 'finalized_model_gender_usman.sav'
    pickle.dump(classifier, open(filename, 'wb'))
    print('Model saved')

    