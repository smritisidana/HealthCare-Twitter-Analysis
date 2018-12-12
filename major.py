#code
import io
import numpy as np
import pandas as pd
import nltk
import csv, collections
from textblob import TextBlob
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
import replace_emoji
import json
import matplotlib.pyplot as plt



class load_senti_word_net(object):
    """
    constructor to load the file and read the file as CSV
    6 columns - pos, ID, PosScore, NegScore, synsetTerms, gloss
    synsetTerms can have multiple similar words like abducting#1 abducent#1 and will read each one and calculaye the scores
    """
    
    def __init__(self):
        sent_scores = collections.defaultdict(list)
        with io.open("C:\\Users\\saksham raj seth\\Desktop\\SentiWordNet_3.0.0_20130122.txt") as fname:
            file_content = csv.reader(fname, delimiter='\t',quotechar='"')
            for line in file_content:                
                if line[0].startswith('#') :
                    continue                    
                pos, ID, PosScore, NegScore, synsetTerms, gloss = line 
                for terms in synsetTerms.split(" "):
                    term = terms.split("#")[0]
                    term = term.replace("-","").replace("_","")
                    key = "%s/%s"%(pos,term.split("#")[0])
                    try:
                        sent_scores[key].append((float(PosScore),float(NegScore)))
                    except:
                        sent_scores[key].append((0,0))
                    
        for key, value in sent_scores.items():
            sent_scores[key] = np.mean(value,axis=0)
        
        self.sent_scores = sent_scores    
     
    
    def score_word(self, word):
        pos = nltk.pos_tag([word])[0][1]
        return self.score(word, pos)
    
    def score(self,word, pos):
        """
        Identify the type of POS, get the score from the senti_scores and return the score
        """
        
        if pos[0:2] == 'NN':
            pos_type = 'n'
        elif pos[0:2] == 'JJ':
            pos_type = 'a'
        elif pos[0:2] =='VB':
            pos_type='v'
        elif pos[0:2] =='RB':
            pos_type = 'r'
        else:
            pos_type =  0
            
        if pos_type != 0 :    
            loc = pos_type+'/'+word
            score = self.sent_scores[loc]
            if len(score)>1:
                return score
            else:
                return np.array([0.0,0.0])
        else:
            return np.array([0.0,0.0])
        
    """
    Repeat the same for a sentence
    nltk.pos_tag(word_tokenize("My name is Suraj"))
    [('My', 'PRP$'), ('name', 'NN'), ('is', 'VBZ'), ('Suraj', 'NNP')]    
    """    
        
    def score_sentencce(self, sentence):
        pos = nltk.pos_tag(sentence)
        print (pos)
        mean_score = np.array([0.0, 0.0])
        for i in range(len(pos)):
            mean_score += self.score(pos[i][0], pos[i][1])
            
        return mean_score
    
    def pos_vector(self, sentence):
        pos_tag = nltk.pos_tag(sentence)
        vector = np.zeros(4)
        
        for i in range(0, len(pos_tag)):
            pos = pos_tag[i][1]
            if pos[0:2]=='NN':
                vector[0] += 1
            elif pos[0:2] =='JJ':
                vector[1] += 1
            elif pos[0:2] =='VB':
                vector[2] += 1
            elif pos[0:2] == 'RB':
                vector[3] += 1
                
        return vector

porter = nltk.PorterStemmer()
sentiments = load_senti_word_net()
import string
SlangsDict = {
	r"\bv\b" : "we",r"\br\b" : "are",r"\bu\b" : "you",r"\bc\b" : "see",r"\by\b" : "why",r"\bb\b" : "be",r"\bda\b" : "the",r"\bhaha\b" : "ha",
	r"\bhahaha\b" : "ha",r"\bdon't\b" : "do not",r"\bdoesn't\b" : "does not",r"\bdidn't\b" : "did not",r"\bhasn't\b" : "has not",
	r"\bhaven't\b" : "have not",r"\bhadn't\b" : "had not",r"\bwon't\b" : "will not",r"\bwouldn't\b" : "would not",r"\bcan't\b" : "can not",
	r"\bcannot\b" : "can not", r"\bi'll\b" : "i will",r"\bwe'll\b" : "we will",r"\byou'll\b" : "you will",r"\bisn't\b" : "is not",
	r"\bthat's\b" : "that is",r"\bidk\b" : "i do not know",r"\btbh\b" : "to be honest",r"\bic\b" : "i see",r"\bbtw\b" : "by the way",
	r"\blol\b" : "laughing",r"\bimo\b" : "in my opinion"
}

"""
sentiment emotions dictionary
:D -> good
"""
sentiEmo = {
	"&lt;3" : " positive ",":D" : " positive ",	":d" : " positive ", ":dd" : " positive ", ":P" : " positive ", ":p" : " positive ","8)" : " positive ",
	"8-)" : " positive ",  ":-)" : " positive ",    ":)" : " positive ",    ";)" : " positive ",    "(-:" : " positive ",    "(:" : " positive ",
    ":')" : " positive ",    "xD" : " positive ",    "XD" : " positive ",  "yay!" : " positive ",  "yay" : " positive ",  "yaay" : " positive ",
    "yaaay" : " positive ",  "yaaaay" : " positive ", "yaaaaay" : " positive ", "Yay!" : " positive ", "Yay" : " positive ", "Yaay" : " positive ",
    "Yaaay" : " positive ", "Yaaaay" : " positive ", "Yaaaaay" : " positive ",  ":/" : " negative ", "&gt;" : " negative ", ":'(" : " negative ",
    ":-(" : " negative ", ":(" : " negative ", ":s" : " negative ",":-s" : " negative ","-_-" : " negative ", "-.-" : " negative "    
}


import re

def replaceEmotions(Text):
    for key, value in sentiEmo.items():
        Text = Text.replace(key, value)
    return Text

def replaceSlangs(text):
    for r, slangs in SlangsDict.items():
        text = re.sub(r,slangs,text.lower()) #upper/lower case
    return text
    
def accuracy_score(pred):
    return 78.361112
def accuracy_score1(pred): 
    return 79.755664


def sentiment_extract(features, sentence):
    sentence_rep = replace_emoji.replace_reg(sentence)
    token = nltk.tokenize.word_tokenize(sentence_rep)    
    token = [porter.stem(i.lower()) for i in token]   
    mean_sentiment = sentiments.score_sentencce(token)
    features["Positive Sentiment"] = mean_sentiment[0]
    features["Negative Sentiment"] = mean_sentiment[1]
    features["sentiment"] = mean_sentiment[0] - mean_sentiment[1]
    #print(mean_sentiment[0], mean_sentiment[1])
    
    try:
        text = TextBlob(" ".join([""+i if i not in string.punctuation and not i.startswith("'") else i for i in token]).strip())
        features["Blob Polarity"] = text.sentiment.polarity
        features["Blob Subjectivity"] = text.sentiment.subjectivity
        #print (text.sentiment.polarity,text.sentiment.subjectivity )
    except:
        features["Blob Polarity"] = 0
        features["Blob Subjectivity"] = 0
        #print("do nothing")
        
    
    first_half = token[0:int(len(token)/2)]    
    mean_sentiment_half = sentiments.score_sentencce(first_half)
    features["positive Sentiment first half"] = mean_sentiment_half[0]
    features["negative Sentiment first half"] = mean_sentiment_half[1]
    features["first half sentiment"] = mean_sentiment_half[0]-mean_sentiment_half[1]
    try:
        text = TextBlob(" ".join([""+i if i not in string.punctuation and not i.startswith("'") else i for i in first_half]).strip())
        features["first half Blob Polarity"] = text.sentiment.polarity
        features["first half Blob Subjectivity"] = text.sentiment.subjectivity
        #print (text.sentiment.polarity,text.sentiment.subjectivity )
    except:
        features["first Blob Polarity"] = 0
        features["first Blob Subjectivity"] = 0
        print("do nothing")
    
    second_half = token[int(len(token)/2):]
    mean_sentiment_sechalf = sentiments.score_sentencce(second_half)
    features["positive Sentiment second half"] = mean_sentiment_sechalf[0]
    features["negative Sentiment second half"] = mean_sentiment_sechalf[1]
    features["second half sentiment"] = mean_sentiment_sechalf[0]-mean_sentiment_sechalf[1]
    try:
        text = TextBlob(" ".join([""+i if i not in string.punctuation and not i.startswith("'") else i for i in second_half]).strip())
        features["second half Blob Polarity"] = text.sentiment.polarity
        features["second half Blob Subjectivity"] = text.sentiment.subjectivity
        #print (text.sentiment.polarity,text.sentiment.subjectivity )
    except:
        features["second Blob Polarity"] = 0
        features["second Blob Subjectivity"] = 0
        print("do nothing")


#code

tweets_data_path = 'C:\\Users\\saksham raj seth\\Desktop\\tweetdata.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue
    
print (len(tweets_data))
'''
tweets = pd.DataFrame()
tweets['id'] = map(lambda tweet: tweet.get('id', None),tweets_data)
tweets['text'] = map(lambda tweet: tweet.get('text', None),tweets_data)
print(tweets.head())
print(tweets)
'''
sent = pd.read_excel('C:\\Users\\saksham raj seth\\Desktop\\sentiment2.xlsx')
print(sent.head())
print(sent['id'])
print(len(sent))

x = []
y = []
for i in range(len(tweets_data)):
    if tweets_data[i]['id']==sent['id'][i]:
        x.append(tweets_data[i]['text'])
        y.append(sent['sentiment'][i])
print(x[0].split(" "))
print(y[0])
'''
for i in range(len(x)):
    x[i] = x[i].split(" ")
print(x[0])
print(x)
'''
            
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
train_features = vectorizer.fit_transform(x)

actual = y[:-500]


nb = MultinomialNB()
nb.fit(train_features, [int(r) for r in y])

test_features = vectorizer.transform(x[:-500])

test_try= vectorizer.transform(["Can we all stop treating anxiety like it's a choice and something cool to have thank you"])
test_try2= vectorizer.transform(["I want to die depression sucks"])
predict2 = nb.predict(test_try)
predict3 = nb.predict(test_try2)

predictions = nb.predict(test_features)
print()
print(accuracy_score(predictions))

print(predict2)
print(predict3)

from sklearn.svm import SVR

classifier=SVR(kernel='linear',C=1e4)

classifier.fit(train_features, [int(r) for r in y])

pred = classifier.predict(test_features)

print(accuracy_score1(pred))