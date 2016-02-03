# coding: utf-8

from question1 import *
import json

'''
helper class to load a thesaurus from disk
input: thesaurusFile, file on disk containing a thesaurus of substitution words for targets
output: the thesaurus, as a mapping from target words to lists of substitution words
'''
def load_thesaurus(thesaurusFile):
	thesaurus = {}
	with open(thesaurusFile) as inFile:
		for line in inFile.readlines():
			word, subs = line.strip().split("\t")
			thesaurus[word] = subs.split(" ")
	return thesaurus

'''
(a) function addition for adding 2 vectors
input: vector1
input: vector2
output: addVector, the resulting vector when adding vector1 and vector2
'''
def addition(vector1, vector2):
	# your code here
	return None

'''
(a) function multiplication for multiplying 2 vectors
input: vector1
input: vector2
output: mulVector, the resulting vector when multiplying vector1 and vector2
'''
def multiplication(vector1, vector2):
	# your code here
	return None

'''
(d) function prob_z_given_w to get probability of LDA topic z, given target word w
input: ldaModel
input: topicID as an integer
input: wordVector in frequency space
output: probability of the topic with topicID in the ldaModel, given the wordVector
'''
def prob_z_given_w(ldaModel, topicID, wordVector):
	# your code here
	return None

'''
(d) function prob_w_given_z to get probability of target word w, given LDA topic z
input: ldaModel
input: targetWord as a string
input: topicID as an integer
output: probability of the targetWord, given the topic with topicID in the ldaModel
'''
def prob_w_given_z(ldaModel, targetWord, topicID):
	# your code here
	return None

'''
get the best substitution word in a given sentence, according to a given model (tf-idf, word2vec, LDA) and type (addition, multiplication, lda)
'''
def best_substitute(jsonSentence, thesaurus, word2id, model, frequencyVectors, csType):
	
	# (b) use addition to get context sensitive vectors
	if csType == "addition":
		# your code here
		pass
		
	# (c) use multiplication to get context sensitive vectors
	elif csType == "multiplication":
		#your code here
		pass
		
	# (d) use LDA to get context sensitive vectors
	elif csType == "lda":
		# your code here
		pass
	
	return None

if __name__ == "__main__":
	import sys
	
	part = sys.argv[1]
	
	# this can give you an indication whether part a (vector addition and multiplication) works.
	if part == "a":
		print("(a): vector addition and multiplication")
		v1, v2, v3 , v4 = [(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)], [1, 0, 1, 0, 2], [1, 2, 0, 0, 1]
		try:
			if not set(addition(v1, v2)) == set([(0, 2), (2, 1), (4, 3), (1, 2)]):
				print("\tError: sparse addition returned wrong result")
			else:
				print("\tPass: sparse addition")
		except Exception as e:
			print("\tError: exception raised in sparse addition")
			print(e)
		try:
			if not set(multiplication(v1, v2)) == set([(0,1), (4,2)]):
				print("\tError: sparse multiplication returned wrong result")
			else:
				print("\tPass: sparse multiplication")
		except Exception as e:
			print("\tError: exception raised in sparse multiplication")
			print(e)
		try:
			addition(v3,v4)
			print("\tPass: full addition")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)
		try:
			multiplication(v3,v4)
			print("\tPass: full multiplication")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)
	
	# you may complete this to get answers for part b (best substitution words with tf-idf and word2vec, using addition)
	if part == "b":
		print("(b) using addition to calculate best substitution words")
		# your code here
	
	# you may complete this to get answers for part c (best substitution words with tf-idf and word2vec, using multiplication)
	if part == "c":
		print("(c) using multiplication to calculate best substitution words")
	
	# this can give you an indication whether your part d1 (P(Z|w) and P(w|Z)) works
	if part == "d":
		print("(d): calculating P(Z|w) and P(w|Z)")
		print("\tloading corpus")
		id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
		print("\tloading LDA model")
		ldaModel = gensim.models.ldamodel.LdaModel.load("lda.model")
		houseTopic = ldaModel[vectors[word2id["house.n"]]][0][0]
		try:
			if prob_z_given_w(ldaModel, houseTopic, vectors[word2id["house.n"]]) > 0.0:
				print("\tPass: P(Z|w)")
			else:
				print("\tFail: P(Z|w)")
		except Exception as e:
			print("\tError: exception during P(Z|w)")
			print(e)
		try:
			if prob_w_given_z(ldaModel, "house.n", houseTopic) > 0.0:
				print("\tPass: P(w|Z)")
			else:
				print("\tFail: P(w|Z)")
		except Exception as e:
			print("\tError: exception during P(w|Z)")
			print(e)
	
	# you may complete this to get answers for part d2 (best substitution words with LDA)
	if part == "e":
		print("(e): using LDA to calculate best substitution words")
		# your code here
