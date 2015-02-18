from __future__ import division
import pandas as pd
import numpy as np


class Unigram_Predict():

	def __init__(self):
		unigram_score = dict()


	def main(self):
		train = pd.read_csv('data/train.tsv', sep='\t')
		train['PhraseLength'] = train['Phrase'].apply(lambda x: len(x.split()))
		almost_dict = train[(train['PhraseLength'] == 1) & (train['Sentiment'] != 2)][['Phrase', 'Sentiment']]
		almost_dict['Phrase'] = almost_dict['Phrase'].apply(lambda x: x.lower())
		self.unigram_score =almost_dict.set_index('Phrase').to_dict()['Sentiment']
		test = pd.read_csv('data/test.tsv', sep='\t')
		test['Sentiment'] = test['Phrase'].apply(lambda x: self.score_phrase(x))
		test.to_csv('data/submission.csv', index=False, columns =['PhraseId', 'Sentiment'])

	def score_phrase(self, phrase):
	    score = 0
	    words = phrase.split()
	    words_counted = 0
	    if len(words) == 0:
	        return 2
	    for word in words:
	        if word in self.unigram_score:
	            words_counted += 1
	            score += self.unigram_score[word]
	    if words_counted == 0:
	        return 2
	    return int(round(score / words_counted))

if __name__ == '__main__':
	uni = Unigram_Predict()
	uni.main()