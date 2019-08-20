import sys
import os
import xml.etree.ElementTree as ET
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#arrays that lists down all names of train and text files
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

#
testdata = []
traindata =[]

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

def compare(id, tweets):
    for line in traindata:
        key,ishuman,gender = line.split(':::')
        if key == id:
            for t in tweets:
                authors_en.append(key)
                t = t.lower()
                t = clean_text(t)
                t = cleanHtml(t)
                t = cleanPunc(t)
                t = keepAlpha(t)
                t = removeStopWords(t)
                t = stemming(t)
                tweets_en.append(t)
                isHuman_en.append(ishuman)
                gender_en.append(gender)
            #for i in range(len(authors_en_test)):
            #    print(authors_en[i],'\t',tweets_en[i],'\t',isHuman_en[i],'\t',gender_en[i])
            break

    for line in testdata:
        key,ishuman,gender = line.split(':::')
        if key == id:
            for t in tweets:
                authors_en_test.append(key)
                t = t.lower()
                t = clean_text(t)
                t = cleanHtml(t)
                t = cleanPunc(t)
                t = keepAlpha(t)
                t = removeStopWords(t)
                t = stemming(t)
                tweets_en_test.append(t)
                isHuman_en_test.append(ishuman)
                gender_en_test.append(gender)
            #for i in range(len(authors_en_test)):
            #    print(authors_en_test[i],'\t',tweets_en_test[i],'\t',isHuman_en_test[i],'\t',gender_en_test[i])
            break
    return

def writeToFile():
    train = open('train.txt','w',encoding="utf-8")
    test = open('test.txt','w',encoding="utf-8")

    for i in range(len(authors_en)):
        cleanString = re.sub('\W+',' ', tweets_en[i] )
        if isHuman_en[i] == 'human':
            if gender_en[i].rstrip() == 'male':
                train.write(authors_en[i]+':::'+cleanString+':::'+'1'+':::'+'1'+'\n')
            else:
                train.write(authors_en[i]+':::'+cleanString+':::'+'1'+':::'+'0'+'\n')
        else:
            train.write(authors_en[i]+':::'+cleanString+':::'+'0'+':::'+'2'+'\n')
    print('Train Done')

    for i in range(len(authors_en_test)):
        cleanString = re.sub('\W+',' ', tweets_en_test[i] )
        if isHuman_en_test[i] == 'human':
            if gender_en[i].rstrip() == 'male':
                test.write(authors_en_test[i]+':::'+cleanString+':::'+'1'+':::'+'1'+'\n')
            else:
                test.write(authors_en_test[i]+':::'+cleanString+':::'+'1'+':::'+'0'+'\n')
        else:
            test.write(authors_en_test[i]+':::'+cleanString+':::'+'0'+':::'+'2'+'\n')
    print('Test Done')
    print('Written to files')
if __name__ == "__main__":
   #input = sys.argv[1]
   #output = sys.argv[2]
   input_ = 'pan19-author-profiling-training-2019-02-18'
   output = 'pan19-author-profiling-output-2019-02-18'

   input_en = input_+'/en/'

   output_en = output+'/en/'

   files = os.listdir(input_en)

   ignore = ['truth-train.txt','truth-dev.txt','truth.txt']

   truthtrain = input_en+'/'+ignore[0]
   truthtest = input_en+'/'+ignore[1]

   f = open(truthtrain)
   g = open(truthtest)

   for line in f: #train       
       traindata.append(line)


   for line in g: #test
       testdata.append(line)

   counter = 0
   for file in files:
       if file not in ignore:
           tweets = []
           tree = ET.parse(input_en+file)  
           root = tree.getroot()
           for child in root:
               for sub in child:
                   tweets.append(sub.text)
           id = file.split('.')[0]
           compare(id,tweets)
           counter+=1
           print(counter,' Processed')

   print(len(authors_en))
   print(len(authors_en_test))

   writeToFile()


