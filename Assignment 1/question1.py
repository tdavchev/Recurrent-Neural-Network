# coding: utf-8

import gensim
import math
import numpy # I included this
from copy import copy
from sets import Set

from itertools import repeat
from collections import defaultdict

'''
(f) helper class, do not modify.
provides an iterator over sentences in the provided BNC corpus
input: corpus path to the BNC corpus
input: n, number of sentences to retrieve (optional, standard -1: all)
'''
class BncSentences:
    def __init__(self, corpus, n=-1):
        self.corpus = corpus
        self.n = n
    
    def __iter__(self):
        n = self.n
        ret = []
        for line in open(self.corpus):
            line = line.strip().lower()
            if line.startswith("<s "):
                ret = []
            elif line.strip() == "</s>":
                if n > 0:
                    n -= 1
                if n == 0:
                    break
                yield copy(ret)
            else:
                parts = line.split("\t")
                if len(parts) == 3:
                    word = parts[-1]
                    idx = word.rfind("-")
                    word, pos = word[:idx], word[idx+1:]
                    if word in ["thus", "late", "often", "only"]:
                        pos = "r"
                    if pos == "j":
                        pos = "a"
                    ret.append(gensim.utils.any2unicode(word + "." + pos))

'''
(a) function load_corpus to read a ciroys from disk
input: vocabFile containing vocabulary
input: contextFile containing word contexts
output: id2word mapping word IDs to words
output: word2id mapping words to word IDs
output: vectors for the corpus, as a list of sparse vectors
'''
def load_corpus(vocabFile, contextFile):
    id2word = {}
    word2id = {}
    vectors = []
    
    fp = open(vocabFile)
    contents = fp.read()
    vocab = contents.split()

    keys = list(range(0,len(vocab)))

    _id2word = zip(keys,vocab)
    _word2id = zip(vocab, keys)

    id2word = dict(_id2word)
    word2id = dict(_word2id)

    cp = open(contextFile)
    _contexts = cp.read()
    contexts = _contexts.split("\n")

    opa = [contexts[i].split() for i in xrange(0, len(contexts))]
    vectorsFast = [[] for i in repeat(None, len(opa))]
    for vector in xrange(0,len(opa)):
        for brei in enumerate(opa[vector]):
            # print brei[0]
            if brei[1].find(":") > -1:
                w,n =  brei[1].split(":")
                vectorsFast[vector].append((w,n))
        # vectorsFast[vector] = dict((vectorsFast[vector][element][0],vectors[vector][element][1]) for element in xrange(0,len(vectors)))

    vectors = vectorsFast
    return id2word, word2id, vectors

'''
(b) function cosine_similarity to calculate similarity between 2 vectors
input: vector1
input: vector2
output: cosine similarity between vector1 and vector2 as a real number
'''
def cosine_similarity(vector1, vector2):
    if vector1==[] or vector2==[]:
        return 0

    v1=vector1
    v2=vector2
    keys1=[]
    keys2=[]
    
    # vector1=map(int,vector1)
    # vector2=map(int,vector2)
    if type(vector1[0])==tuple:
        #convert to dictionary
        dictionary1 = dict((x, y) for x, y in vector1)
        vector1=dictionary1.values() #get all values for the Euclidean distance
        vector1 = map(int, vector1) # provide itemsize for data type
        v1=[] # will be recomuputed
        keys1 = dictionary1.keys() # we need all keys to be able to compare
    if type(vector2[0])==tuple:
        #convert to dictionary
        dictionary2 = dict((x, y) for x, y in vector2)
        vector2=dictionary2.values() #get all values for the Euclidean distance
        vector2 = map(int, vector2) # provide itemsize for data type
        v2=[] # will be recomputed
        keys2 = dictionary2.keys() # we need all keys to be able to compare

    # handle cases when only one of the vectors is sparse:
    if keys1==[]:
        keys=keys2
    else:
        keys=keys1

    for key in keys:
        # handle cases when only one of the vectors is sparse:
        if keys2==[]:
            keys_ = list(range(0,len(v2)))
        elif keys1==[]:
            keys_ = list(range(0,len(v1)))
        else:
            keys_ = keys2
        # If the key is in only one of the vectors then it is not needed
        # because it will be multiplied by 0
        if key in keys_: # consider only values which indexes appear in both vectors
            v1.append(dictionary1[key])
            v2.append(dictionary2[key])

    # provide itemsize for data type
    v1 = map(int,v1)
    v2 = map(int,v2)

    return numpy.dot(v1,v2)/numpy.dot(numpy.linalg.norm(vector1),numpy.linalg.norm(vector2))

'''
(d) function tf_idf to turn existing frequency-based vector model into tf-idf-based vector model
input: freqVectors, a list of frequency-based vectors
output: tfIdfVectors, a list of tf-idf-based vectors
'''
def tf_idf(freqVectors):
    #print freqVectors[0]
    # z = []
    # print len(freqVectors)
    tfIdfVectors = []
    # nDocs=len(freqVectors)
    # compute how many times one finds a file
    lista = []
    N = len(freqVectors)
    print "begin..."
    # for vector in xrange(0,len(freqVectors)):
    #     for word in xrange(0,len(freqVectors[vector])):
    #         if int(freqVectors[vector][word][0]):

    #             print "00000000000000000000000"
    #             print vector
    #             print "**********************"
    #             print freqVectors[vector][word]
    #             print "!!!!!!!!!!!!!!!!!!!!!!"
    #             print int(freqVectors[vector][word][0])
    
    lista=[int(freqVectors[vector][word][0]) for vector in xrange(0,len(freqVectors)) for word in xrange(0,len(freqVectors[vector]))]
    lista.sort()
    # print lista
    print "list with counts gathered"
    #print lista
    wordFreq = []
    idx = ""
    prevWord = ""
    count = 0
    countedDict = {}
    for word in lista:
        if idx == "":
            idx = word
        if word == idx:
            count = count + 1
        else: # the word has changed
            prevWord = idx
            countedDict[idx]=count
            idx = word
            count = 1
    if idx not in countedDict.keys():
        countedDict[idx]=count
    # wordFreq = [lista.count(w) for w in lista]
    # print wordFreq
    # nT =  zip(lista,wordFreq)
    print "nT computed;"
    N=3 
    tf_idf = [[] for i in repeat(None, len(freqVectors))]

    for vector in xrange(0,len(freqVectors)):
        for item in xrange(0,len(freqVectors[vector])):
            word = freqVectors[vector][item][0]
            freq = freqVectors[vector][item][1]
            tf_ = 1+numpy.log2(float(freq))
            _idf = numpy.log2(float(N)/float((1+countedDict[int(word)])))
            tf_idf[vector].append((word,tf_*_idf))
    print "tf_idf computed"
    
    print "removing all zeros;"
    for j in xrange(0,len(tf_idf)):
        tf_idf[j] = filter(lambda a: a != 0, tf_idf[j])

    tfIdfVectors = tf_idf
    # # for vector in xrange(0, len(freqVectors)):
    # tf = [float(freqVectors[vector][j][1]) for vector in xrange(0,len(freqVectors)) for j in xrange(0,len(freqVectors[vector]))] # celiq spisuk s sizeove
    # words = [freqVectors[vector][j][0] for vector in xrange(0,len(freqVectors)) for j in xrange(0,len(freqVectors[vector]))]
    # print "tf computed"
    # lengths = [len(freqVectors[vector]) for vector in xrange(0,len(freqVectors))] # broi elementi ot kude do kude znaesh e vseki vector
    # '''
    # Computing lengths from vectors 0 to 5 results in --> [4999, 4593, 4602, 3793, 4449] but we want a 0 in front for the tf_ computation
    # '''
    # lengths.insert(0,0)
    # print "lengths computed"
    # tf_ = [[] for i in repeat(None, len(lengths))]
    # idf_ = [[] for i in repeat(None, len(lengths))]
    # for document in xrange(1,len(lengths)):
    #     for idx,_tf in enumerate(tf[lengths[document-1]:lengths[document]]):
    #         tf_[document-1].append(float(1+numpy.log2(_tf)))
    #         idf_[document-1].append(float((numpy.log2(float(N)/float(words[idx])))))
    
    # print type(tf_)
    # print type(idf_)
    # # tf_idf = numpy.zeros()
    # # tf_idf = numpy.asarray(tf_idf)
    # # tf_idf = numpy.dot(a,b)
    # # print len(tf_idf[0])
    # print tf_idf
    # print len(idf_[0])
    # print len(tf_[0]) # results in 4999
    # _idf = 

    return tfIdfVectors

'''
(f) function word2vec to build a word2vec vector model with 100 dimensions and window size 5
'''
def word2vec(corpus, learningRate, downsampleRate, negSampling):
    # your code here
    return None

'''
(h) function lda to build an LDA model with 100 topics from a frequency vector space
input: vectors
input: wordMapping (optional) mapping from word IDs to words
output: an LDA topic model with 100 topics, using the frequency vectors
'''
def lda(vectors, wordMapping):
    # your code here
    return None

'''
(j) function get_topic_words, to get words in a given LDA topic
input: ldaModel, pre-trained Gensim LDA model
input: topicID, ID of the topic for which to get topic words
input: wordMapping, mapping from words to IDs (optional)
'''
def get_topic_words(ldaModel, topicID):
    # your code here
    return None

if __name__ == '__main__':
    import sys
    
    part = sys.argv[1].lower()
    
    # these are indices for house, home and time in the data. Don't change.
    house_noun = 80
    home_noun = 143
    time_noun = 12
    
    # this can give you an indication whether part a (loading a corpus) works.
    # not guaranteed that everything works.
    if part == "a":
        print("(a): load corpus")
        try:
            id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
            print len(vectors)
            if not id2word:
                print("\tError: id2word is None or empty")
                exit()
            if not word2id:
                print("\tError: id2word is None or empty")
                exit()
            if not vectors:
                print("\tError: id2word is None or empty")
                exit()
            print("\tPass: load corpus from file")
        except Exception as e:
            print("\tError: could not load corpus from disk")
            print(e)
        
        try:
            if not id2word[house_noun] == "house.n" or not id2word[home_noun] == "home.n" or not id2word[time_noun] == "time.n":
                print("\tError: id2word fails to retrive correct words for ids")
            else:
                print("\tPass: id2word")
        except Exception:
            print("\tError: Exception in id2word")
            print(e)
        
        try:
            if not word2id["house.n"] == house_noun or not word2id["home.n"] == home_noun or not word2id["time.n"] == time_noun:
                print("\tError: word2id fails to retrive correct ids for words")
            else:
                print("\tPass: word2id")
        except Exception:
            print("\tError: Exception in word2id")
            print(e)
    
    # this can give you an indication whether part b (cosine similarity) works.
    # these are very simple dummy vectors, no guarantee it works for our actual vectors.
    if part == "b":
        import numpy
        print("(b): cosine similarity")
        try:
            cos = cosine_similarity([(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)])
            if not numpy.isclose(0.5, cos):
                print("\tError: sparse expected similarity is 0.5, was {0}".format(cos))
            else:
                print("\tPass: sparse vector similarity")
        except Exception:
            print("\tError: failed for sparse vector")
        try:
            cos = cosine_similarity([1, 0, 1, 0, 2], [1, 2, 0, 0, 1])
            if not numpy.isclose(0.5, cos):
                print("\tError: full expected similarity is 0.5, was {0}".format(cos))
            else:
                print("\tPass: full vector similarity")
        except Exception:
            print("\tError: failed for full vector")

    # you may complete this part to get answers for part c (similarity in frequency space)
    if part == "c":
        print("(c) similarity of house, home and time in frequency space")
        houseVector=[]
        homeVector=[]
        timeVector=[]
        # try:
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        for key in xrange(0,len(vectors)):
            for element in xrange(0,len(vectors[key])):
                if vectors[key][element][0]=='80':
                    houseVector.append((key,vectors[key][element][1]))
                if vectors[key][element][0]=='143':
                    homeVector.append((key,vectors[key][element][1]))
                if vectors[key][element][0]=='12':
                    timeVector.append((key,vectors[key][element][1]))
        cos = cosine_similarity(houseVector, homeVector)
        cos2 = cosine_similarity(timeVector, homeVector)
        cos3 = cosine_similarity(houseVector, timeVector)
        print cos
        print cos2
        print cos3
        # except Exception as e:
        #     print("\tError: Exception in similarity of house,home and time")
        #     print(e)
        try:
            if not word2id["house.n"] == house_noun or not word2id["home.n"] == home_noun or not word2id["time.n"] == time_noun:
                print("\tError: word2id fails to retrive correct ids for words")
            else:
                print("\tPass: word2id")
        except Exception:
            print("\tError: Exception in word2id")
            print(e)
    
    
    # this gives you an indication whether your conversion into tf-idf space works.
    # this does not test for vector values in tf-idf space, hence can't tell you whether tf-idf has been implemented correctly
    if part == "d":
        print("(d) converting to tf-idf space")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        try:
            tfIdfSpace = tf_idf(vectors) #must pass a [vector[0]] in case of 1 vector
            # print tfIdfSpace
            if not len(vectors) == len(tfIdfSpace):
                print("\tError: tf-idf space does not correspond to original vector space")
            else:
                print("\tPass: converted to tf-idf space")
        except Exception as e:
            print("\tError: could not convert to tf-idf space")
            print(e)
    
    # you may complete this part to get answers for part e (similarity in tf-idf space)
    if part == "e":
        print("(e) similarity of house, home and time in tf-idf space")
        # your code here
        _, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        tfIdfSpace = tf_idf(vectors)
        print("similarity of house and home is %s" % (cosine_similarity(tfIdfSpace[word2id["house.n"]], tfIdfSpace[word2id["home.n"]])))
        print("similarity of house and time is %s" % (cosine_similarity(tfIdfSpace[word2id["house.n"]], tfIdfSpace[word2id["time.n"]])))
        print("similarity of home and time is %s" % (cosine_similarity(tfIdfSpace[word2id["home.n"]], tfIdfSpace[word2id["time.n"]])))
    
    # you may complete this part for the first part of f (estimating best learning rate, sample rate and negative samplings)
    if part == "f1":
        print("(f1) word2vec, estimating best learning rate, sample rate, negative sampling")
        
        # your code here
    
    # you may complete this part for the second part of f (training and saving the actual word2vec model)
    if part == "f2":
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        print("(f2) word2vec, building full model with best parameters. May take a while.")
        
        # your code here
    
    # you may complete this part to get answers for part g (similarity in your word2vec model)
    if part == "g":
        print("(g): word2vec based similarity")
        
        # your code here
    
    # you may complete this for part h (training and saving the LDA model)
    if part == "h":
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        print("(h) LDA model")
        
        # your code here
    
    # you may complete this part to get answers for part i (similarity in your LDA model)
    if part == "i":
        print("(i): lda-based similarity")
        
        # your code here

    # you may complete this part to get answers for part j (topic words in your LDA model)
    if part == "j":
        print("(j) get topics from LDA model")
        
        # your code here
