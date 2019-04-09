import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords as stopwordloader
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import re
import emoticons

nltk.download()

df = pd.read_csv(r'train/downloaded.tsv', sep='\t', header=None)
# print(df)

stopwords = stopwordloader.words('english')
# puntuaion we choose to ignore, keeping '?' and '!'
punc = set('"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~')

def data_normalization(dataset):
    # delete not available tweets from training_set
    dataset = dataset[dataset[3] != 'Not Available']
    # changing labels that mean neutral but in the dataset they have a different title
    dataset = dataset.replace('objective', 'neutral')
    dataset = dataset.replace('objective-OR-neutral', 'neutral')

    for index, row in dataset.iterrows():
        # substitute numbers for the word 'NUMBER'
        token = [u'NUMBER' if re.match(r"\d", x) else x for x in row[3]]
        print(token)

def read_SentiStrengh_dict():
    dictionary = {}

    # read emotion list
    fp = open("dataset/dictionary/EmotionLookupTable.txt");
    for line in fp:
        line = line.split('\t')
        if len(line) >= 2:
            word = line[0].strip()
            #word = word.decode()
            so = line[1].strip()
            dictionary[word] = float(so)
    fp.close()

    # read emoticons list
    fp = open("dataset/dictionary//EmoticonLookupTable.txt", encoding = "ISO-8859-1");
    for line in fp:
        line = line.split('\t')
        if len(line) >= 2:
            word = line[0].strip()
            #word = word.decode(errors='replace')
            so = line[1].strip()
            dictionary[word] = float(so)
    fp.close()

    return dictionary

def find_word(word, word_list):

    result = None
    matches = [w for w in sorted(word_list) if (w == word) or (w[-1] == '*' and word.startswith(w[:-1]))]
    longest_match = 0
    for match in matches:
        if len(match) > longest_match:
            longest_match = len(match)
            result = match

    return result

dictionay = read_SentiStrengh_dict()


def tokanizeTraining(training_set):
    tknzr = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)

    # delete not available tweets from training_set
    training_set = training_set[training_set[3] != 'Not Available']

    training_set = training_set.replace('objective', 'neutral')
    training_set = training_set.replace('objective-OR-neutral', 'neutral')

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
        for symbol in emoticons.emoticons:
            token = re.sub(r'(' + re.escape(symbol) + r')[^a-z0-9A-Z]', ' \g<1> ' + emoticons.emoticons[symbol] + ' ',
                                   token + ' ')
            print(token)
        token = [x.lower() if not x.isupper() else x for x in token]
        token = [WordNetLemmatizer().lemmatize(x) if not x.isupper() else x for x in token]
        token = [x for x in token if not x in stopwords]
        #print(token)
        #tokenList.append(nltk.pos_tag(token))
        tokenList.append(token)
        y.append(row[2])

    return tokenList, y


token_list, Y = tokanizeTraining(df)
