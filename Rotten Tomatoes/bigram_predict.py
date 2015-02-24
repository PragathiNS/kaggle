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

def score_phrase(phrase, score = 0, total_found = 0):
    if len(phrase) == 0:
        return 2 if total_found ==0 else score / total_found
    if len(phrase) == 1:
        temp_score = unigram_score_clean.get(phrase[0], 2)
        if temp_score <= 1.5 or temp_score >= 2.5:
            return (score + temp_score) / (total_found + 1)
        else:
            return 2 if total_found ==0 else score / total_found
    if len(phrase) == 2:
        if phrase in bigram_dict:
            temp_score = bigram_dict[phrase]
            if temp_score <= 1.5 or temp_score >= 2.5:
                return (score + temp_score) / (total_found + 1)
            else:
                return 2 if total_found == 0 else score / total_found
        else:
            temp_score = unigram_score_clean.get(phrase[0], 2)
            if temp_score <= 1.5 or temp_score >= 2.5:
                return score_phrase(phrase[1:], score + temp_score, total_found + 1)
            else:
                return score_phrase(phrase[1:], score, total_found)
    else:
        if phrase[:2] in bigram_dict:
            temp_score = bigram_dict[phrase[:2]]
            if temp_score <= 1.5 or temp_score >= 2.5:
                return score_phrase(phrase[2:], score + temp_score, total_found + 1)
            else:
                return score_phrase(phrase[2:], score, total_found)
        elif phrase[0] in unigram_score_clean:
            temp_score = unigram_score_clean[phrase[0]]
            if temp_score <= 1.5 or temp_score >= 2.5:
                return score_phrase(phrase[1:], score + temp_score, total_found + 1)
            else:
                return score_phrase(phrase[1:], score, total_found)
        else:
            return score_phrase(phrase[1:], score, total_found)   

train = pd.read_csv('data/train.tsv', sep='\t')
train['clean_phrase']=train['Phrase'].apply(clean_phrase)
train['PhraseLength'] = train['clean_phrase'].apply(lambda x: len(x.split()))
almost_bigram_dict = train[train['PhraseLength'] == 2][['clean_phrase', 'Sentiment']]
almost_unigram_dict = train[train['PhraseLength'] == 1][['clean_phrase', 'Sentiment']]
unigram_score_clean = almost_unigram_dict.groupby('clean_phrase')['Sentiment'].mean().to_dict()
bigram_dict = dict()
for row in pd.DataFrame(almost_bigram_dict.groupby('clean_phrase')['Sentiment'].mean()).reset_index().values:
    bigram_dict[tuple(row[0].split())] = row[1]

test = pd.read_csv('data/test.tsv', sep='\t')
test['clean_phrase']=test['Phrase'].apply(clean_phrase)
test['bigram_score'] = test['clean_phrase'].apply(lambda x: int(round(score_phrase(tuple(x.split())))))
test[['PhraseId', 'bigram_score']].rename(columns={'bigram_score' : 'Sentiment'}).to_csv('data/bigram_predictor.csv', index=False)