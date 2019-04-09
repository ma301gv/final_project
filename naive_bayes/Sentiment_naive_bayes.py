import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords as stopwordloader
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import string

# nltk.download()

df = pd.read_csv(r'train/downloaded.tsv', sep='\t', header=None)
# print(df)



# puntuaion we choose to ignore, we keep '?' and '!'


def data_preprocessing(training_set):
    tknzr = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)
    stopwords = set(stopwordloader.words('english'))
    punc = set('"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~')

    # delete not available tweets from training_set
    training_set = training_set[training_set[3] != 'Not Available']
    # substitute
    training_set = training_set.replace('objective', 'neutral')
    training_set = training_set.replace('objective-OR-neutral', 'neutral')

    # list of tokenised tweets
    tokenList = []
    # list of labels
    y = []

    ps = PorterStemmer()
    lmtz = WordNetLemmatizer()
    for index, row in training_set.iterrows():
        tweet = row[3].lower()
        tweet = tweet.strip()
        # tokenizing the tweet
        tweet = (tknzr.tokenize(tweet.encode('utf-8')))
        # Stop word removal
        tweet = [i for i in tweet if not i in stopwords]
        # Swap numbers for the word NUMBER
        tweet = [u'NUMBER' if re.match(r"\d", x) else x for x in tweet]
        # Remove @user from tweet
        tweet = [u'AT_USER' if re.match(r"@.*", x) else x for x in tweet]
        # Remove links starting with http
        tweet = [u'LINK' if re.match(r"http.*", x) else x for x in tweet]
        # remove links starting with www
        tweet = [u'LINK' if re.match(r"www.*", x) else x for x in tweet]
        # remove characters leaving '!' and '?'
        tweet = [i for i in tweet if not i in punc]
        # Stemming process
        tweet = [ps.stem(i) for i in tweet]
        # Lemmatisation
        tweet = [lmtz.lemmatize(i) for i in tweet]
        # Part-of-Speech Tagging
        tweet = nltk.pos_tag(tweet)
        #print(tweet)
        tokenList.append(tweet)
    #print(tokenList)
    return tokenList

token_list = data_preprocessing(df)
