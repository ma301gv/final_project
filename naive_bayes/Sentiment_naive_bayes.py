import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords as stopwordloader
from nltk.tokenize import TweetTokenizer
import re

# nltk.download()

df = pd.read_csv(r'train/downloaded.tsv', sep='\t', header=None)
# print(df)

stopwords = stopwordloader.words('english')

# puntuaion we choose to ignore, we keep '?' and '!'
punc = set('"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~')



def tokanizeTraining(training_set):
    tknzr = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)

    # delete not available tweets from training_set
    training_set = training_set[training_set[3] != 'Not Available']

    # list of tokenised tweets
    tokenList = []

    # list of labels
    y = []

    for index, row in training_set.iterrows():
        token = (tknzr.tokenize(row[3].encode('utf-8')))

        token = filter(lambda word: word not in punc, token)
        token = filter(lambda t: t not in stopwords, token)
        # replace numbers, link, @
        token = [u'LINK' if re.match(r"http.*", x) else x for x in token]
        token = [u'AT_USER' if re.match(r"@.*", x) else x for x in token]
        token = [u'NUMBER' if re.match(r"\d", x) else x for x in token]
        token = [x.lower() if not x.isupper() else x for x in token]
        print(token)
        tokenList.append(nltk.pos_tag(token))

        y.append(row[2])


    return tokenList, y

token_list, Y = tokanizeTraining(df)

#print(token_list)