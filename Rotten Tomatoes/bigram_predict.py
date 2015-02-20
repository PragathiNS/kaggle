from __future__ import division
import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


stops =  stopwords.words('english')
stops.remove('not')
stops.remove('but')
porter = PorterStemmer()

def clean_phrase(phrase):
	phrase = re.sub("[^a-zA-Z]",' ',  phrase.lower())
	phrase = re.sub(" . |^. | .$",' ',  phrase.lower())
	phrase = ' '.join([porter.stem(word) for word in phrase.split() if word not in stops])
	return phrase

def score_phrase(phrase, score = 0):
    if len(phrase) == 1:
        return score + unigram_score_clean.get(phrase[0], 0)
    if len(phrase) == 2:
        if phrase in bigram_dict:
            return score + bigram_dict[tuple(phrase)]
        else:
            return score_phrase(phrase[1:], score)
    else:
        if phrase[:2] in bigram_dict:
            return score_phrase(phrase[2:], score + bigram_dict.get(tuple(phrase[:2]), 0))
        elif phrase[0] in unigram_score_clean:
            return score_phrase(phrase[1:], score + unigram_score_clean.get(phrase[0], 0))
        else:
            return score_phrase(phrase[2:], score)

train = pd.read_csv('data/train.tsv', sep='\t')
train['clean_phrase']=train['Phrase'].apply(clean_phrase)
train['PhraseLength'] = train['clean_phrase'].apply(lambda x: len(x.split()))
almost_bigram_dict = train[train['PhraseLength'] == 2][['clean_phrase', 'Sentiment']]
almost_unigram_dict = train[train['PhraseLength'] == 1][['clean_phrase', 'Sentiment']]
bigram_dict = dict()
for row in pd.DataFrame(almost_bigram_dict.groupby('clean_phrase')['Sentiment'].mean()).reset_index().values:
	bigram_dict[tuple(row[0].split())] = row[1]
unigram_score_clean = almost_unigram_dict.groupby('clean_phrase')['Sentiment'].mean().to_dict()

print bigram_dict