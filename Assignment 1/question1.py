# coding: utf-8

import gensim
import math
import numpy # I included this
from copy import copy
import logging
from sets import Set

from itertools import repeat
from collections import defaultdict
logging.basicConfig(filename='task-f02092016.log',format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

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
                    if word in ['thus', 'late', 'often', 'only', 'usually', 'however', 'lately', 'absolutely', 'hardly', 'fairly', 'near', 'similarly', 'sooner', 'there', 'seriously', 'consequently', 'recently', 'across', 'softly', 'together', 'obviously', 'slightly', 'instantly', 'well', 'therefore', 'solely', 'intimately', 'correctly', 'roughly', 'truly', 'briefly', 'clearly', 'effectively', 'sometimes', 'everywhere', 'somewhat', 'behind', 'heavily', 'indeed', 'sufficiently', 'abruptly', 'narrowly', 'frequently', 'lightly', 'likewise', 'utterly', 'now', 'previously', 'barely', 'seemingly', 'along', 'equally', 'so', 'below', 'apart', 'rather', 'already', 'underneath', 'currently', 'here', 'quite', 'regularly', 'elsewhere', 'today', 'still', 'continuously', 'yet', 'virtually', 'of', 'exclusively', 'right', 'forward', 'properly', 'instead', 'this', 'immediately', 'nowadays', 'around', 'perfectly', 'reasonably', 'much', 'nevertheless', 'intently', 'forth', 'significantly', 'merely', 'repeatedly', 'soon', 'closely', 'shortly', 'accordingly', 'badly', 'formerly', 'alternatively', 'hard', 'hence', 'nearly', 'honestly', 'wholly', 'commonly', 'completely', 'perhaps', 'carefully', 'possibly', 'quietly', 'out', 'really', 'close', 'strongly', 'fiercely', 'strictly', 'jointly', 'earlier', 'round', 'as', 'definitely', 'purely', 'little', 'initially', 'ahead', 'occasionally', 'totally', 'severely', 'maybe', 'evidently', 'before', 'later', 'apparently', 'actually', 'onwards', 'almost', 'tightly', 'practically', 'extremely', 'just', 'accurately', 'entirely', 'faintly', 'away', 'since', 'genuinely', 'neatly', 'directly', 'potentially', 'presently', 'approximately', 'very', 'forwards', 'aside', 'that', 'hitherto', 'beforehand', 'fully', 'firmly', 'generally', 'altogether', 'gently', 'about', 'exceptionally', 'exactly', 'straight', 'on', 'off', 'ever', 'also', 'sharply', 'violently', 'undoubtedly', 'more', 'over', 'quickly', 'plainly', 'necessarily']:
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
            if brei[1].find(":") > -1:
                w,n =  brei[1].split(":")
                vectorsFast[vector].append((int(w),int(n)))

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
    # print vector1
    # print vector2
    v1=vector1
    v2=vector2
    keys1=[]
    keys2=[]
    if type(vector1[0])==tuple:
        #convert to dictionary
        dictionary1 = dict((x, y) for x, y in vector1)
        vector1=dictionary1.values() #get all values for the Euclidean distance
        if type(vector2[0])==list:
            vector2=vector2[0]
        vector1 = map(float, vector1) # provide itemsize for data type
        v1=[] # will be recomuputed
        keys1 = dictionary1.keys() # we need all keys to be able to compare
    if type(vector2[0])==tuple:
        #convert to dictionary
        dictionary2 = dict((x, y) for x, y in vector2)
        vector2=dictionary2.values() #get all values for the Euclidean distance
        vector2 = numpy.asarray(vector2)
        vector2=numpy.ndarray.tolist(vector2)
        if type(vector2[0])==list:
            vector2=vector2[0]
        vector2 = map(float, vector2) # provide itemsize for data type
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

    try:
        return numpy.dot(v1,v2)/numpy.dot(numpy.linalg.norm(vector1),numpy.linalg.norm(vector2))
    except Exception, e:
        return -1


'''
(d) function tf_idf to turn existing frequency-based vector model into tf-idf-based vector model
input: freqVectors, a list of frequency-based vectors
output: tfIdfVectors, a list of tf-idf-based vectors
'''
def tf_idf(freqVectors):
    tfIdfVectors = []
    # compute how many times one finds a file
    lista = []
    N = len(freqVectors)
    print "begin..."
    lista=[int(freqVectors[vector][word][0]) for vector in xrange(0,len(freqVectors)) for word in xrange(0,len(freqVectors[vector]))]
    lista.sort()
    print "list with counts gathered"
    wordFreq = []
    idx = ""
    prevWord = ""
    count = 0
    countedDict = {}
    for word in lista:
        if idx == "":
            idx = word
        elif word == idx:
            count = count + 1
        else: # the word has changed
            prevWord = idx
            countedDict[idx]=count
            idx = word
            count = 1
    if idx not in countedDict.keys():
        countedDict[idx]=count

    print "nT computed;"
    tf_idf = [[] for i in repeat(None, len(freqVectors))]

    for vector in xrange(0,len(freqVectors)):
        for item in xrange(0,len(freqVectors[vector])):
            word = freqVectors[vector][item][0]
            freq = freqVectors[vector][item][1]
            tf_ = 1.0+float(numpy.log2(float(freq)))
            div = float(N)
            denom=(1.0+float(countedDict[int(word)]))
            _idf = numpy.log2(div/denom)
            tf_idf[vector].append((word,tf_*_idf))
    print "tf_idf computed"

    for j in xrange(0,len(tf_idf)):
        tf_idf[j] = filter(lambda a: a != 0, tf_idf[j])
    print "all zeros were removed;"

    tfIdfVectors = tf_idf

    return tfIdfVectors

'''
(f) function word2vec to build a word2vec vector model with 100 dimensions and window size 5
'''
def word2vec(corpus, learningRate, downsampleRate, negSampling,n,name='vector.bin'):
    sntncs = BncSentences(corpus, n=n)
    model = gensim.models.Word2Vec(sentences=sntncs, size=100, window=5, alpha=learningRate, negative=negSampling, sg=1, workers=8, sample=downsampleRate)
    model.init_sims(replace=True)
    model.save(name)
    return model

'''
A function to test the accuracy
'''
def test_accuracy(test_set, model):
    acc=model.accuracy(test_set)
    return acc

'''
(h) function lda to build an LDA model with 100 topics from a frequency vector space
input: vectors
input: wordMapping (optional) mapping from word IDs to words
output: an LDA topic model with 100 topics, using the frequency vectors
'''
def lda(vectors, wordMapping=None):
    print "Starting lda"
    lda = gensim.models.ldamodel.LdaModel(vectors, id2word=wordMapping, num_topics=100, update_every=0, passes=10)
    print "Lda completed, saving.."
    lda.save('lda_model_v2', ignore=['state', 'dispatcher'])
    print "Model saved. Showing topics.."
    print lda.show_topics(num_topics=10, num_words=10, log=False, formatted=True)
    print "End."

'''
(j) function get_topic_words, to get words in a given LDA topic
input: ldaModel, pre-trained Gensim LDA model
input: topicID, ID of the topic for which to get topic words
input: wordMapping, mapping from words to IDs (optional)
'''
def get_topic_words(ldaModel, topicID, topn=10):
    return ldaModel.show_topic(topicID,topn)

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
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        for key in xrange(0,len(vectors)):
            for element in xrange(0,len(vectors[key])):
                if vectors[key][element][0]==80:
                    houseVector.append((key,vectors[key][element][1]))
                if vectors[key][element][0]==143:
                    homeVector.append((key,vectors[key][element][1]))
                if vectors[key][element][0]==12:
                    timeVector.append((key,vectors[key][element][1]))
        cos = cosine_similarity(houseVector, homeVector)
        cos2 = cosine_similarity(timeVector, homeVector)
        cos3 = cosine_similarity(houseVector, timeVector)
        print("cosine similarity of house and home is {0}".format(cos))
        print("cosine similarity of time and home is {0}".format(cos2))
        print("cosine similarity of time and house is {0}".format(cos3))
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
            if not len(vectors) == len(tfIdfSpace):
                print("\tError: tf-idf space does not correspond to original vector space")
            else:
                print("\tPass: converted to tf-idf space")
            if not numpy.isclose(29.3685342569, tfIdfSpace[0][0][1]): #precision check
                print("\tError: tf-idf has miscomputed")
            else:
                print("\tPass: output from tf-idf is correct")
        except Exception as e:
            print("\tError: could not convert to tf-idf space")
            print(e)

    # you may complete this part to get answers for part e (similarity in tf-idf space)
    if part == "e":
        print("(e) similarity of house, home and time in tf-idf space")
        # your code here
        _, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        tfIdfSpace = tf_idf(vectors)

        print("similarity of house and home is {0}".format(cosine_similarity(tfIdfSpace[word2id["house.n"]], tfIdfSpace[word2id["home.n"]])))
        print("similarity of house and time is {0}".format(cosine_similarity(tfIdfSpace[word2id["house.n"]], tfIdfSpace[word2id["time.n"]])))
        print("similarity of home and time is {0}".format(cosine_similarity(tfIdfSpace[word2id["home.n"]], tfIdfSpace[word2id["time.n"]])))

    # you may complete this part for the first part of f (estimating best learning rate, sample rate and negative samplings)
    if part == "f1":
        # Total: 7.0% (243/3476)
        lRate=0.05
        sRate=0.001
        nSampling=10
        print("(f1) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,100000)
        print("(f1) word2vec, testing model...")
        acc=test_accuracy(sys.argv[3],w2v)

        # Total 8.0%
        lRate=0.05
        sRate=0.01
        nSampling=10
        print("(f1) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,100000)
        print("(f1) word2vec, testing model...")
        acc=test_accuracy(sys.argv[3],w2v)


        # Total 3.0%
        # lRate=0.01
        # sRate=0.01
        # nSampling=5
        # print("(f1) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        # w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,100000)
        # print("(f1) word2vec, testing model...")
        # acc=test_accuracy(sys.argv[3],w2v)

        # # Total 7.6%
        # lRate=0.03
        # sRate=0.01
        # nSampling=5
        # logging.info("(f1) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        # w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,100000)
        # logging.info("(f1) word2vec, testing model...")
        # acc=test_accuracy(sys.argv[3],w2v)

        # # what happens if we ttry the final values suggested in paper
        # lRate=0.05
        # sRate=0.00001
        # nSampling=10
        # logging.info("(f1) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        # w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,100000)
        # logging.info("(f1) word2vec, testing model...")
        # acc=test_accuracy(sys.argv[3],w2v)

        # # apparently this shows good performance Total 7.7%
        # lRate=0.05
        # sRate=0.01
        # nSampling=10
        # logging.info("(f1) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        # w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,100000)
        # logging.info("(f1) word2vec, testing model...")
        # acc=test_accuracy(sys.argv[3],w2v)

        # #full vector training will happen on those total 8%
        # lRate=0.03
        # sRate=0.01
        # nSampling=10
        # logging.info("(f1) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        # w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,100000)
        # logging.info("(f1) word2vec, testing model...")
        # acc=test_accuracy(sys.argv[3],w2v)

        # # total: 0.0% (1/3476)
        # lRate=0.01
        # sRate=0.01
        # nSampling=0
        # logging.info("(f1) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        # w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,100000)
        # logging.info("(f1) word2vec, testing model...")
        # acc=test_accuracy(sys.argv[3],w2v)

        # # total: 3.6% (124/3476)
        # lRate=0.01
        # sRate=0.01
        # nSampling=10
        # logging.info("(f1) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        # w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,100000)
        # logging.info("(f1) word2vec, testing model...")
        # acc=test_accuracy(sys.argv[3],w2v)

        # # total: 0.4% (14/3476)
        # lRate=0.01
        # sRate=0.0001
        # nSampling=5
        # print("(f1) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        # w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,sys.argv[3])
        # print("(f1) word2vec, testing model...")
        # acc=test_accuracy(sys.argv[3],w2v)

    # you may complete this part for the second part of f (training and saving the actual word2vec model)
    if part == "f2":
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        print("(f2) word2vec, building full model with best parameters. May take a while.")
        lRate=0.05
        sRate=0.01
        nSampling=10
        logging.info("(f2) word2vec, estimating best learning rate {0}, sample rate {1}, negative sampling {2}".format(lRate,sRate,nSampling))
        w2v=word2vec(sys.argv[2],lRate,sRate,nSampling,-1,"final-model-temp09022016.bin")
        logging.info("(f2) word2vec, testing model...")
        acc=test_accuracy(sys.argv[3],w2v)

    # you may complete this part to get answers for part g (similarity in your word2vec model)
    if part == "g":
        print("(g): word2vec based similarity")

        w2v=gensim.models.Word2Vec.load('final-model.bin')
        houseHome = w2v.similarity('house.n', 'home.n')
        houseTime = w2v.similarity('house.n', 'time.n')
        homeTime = w2v.similarity('time.n', 'home.n')

        print("similarity of house and home is {0}".format(houseHome))
        print("similarity of house and time is {0}".format(houseTime))
        print("similarity of home and time is {0}".format(homeTime))

    # you may complete this for part h (training and saving the LDA model)
    if part == "h":
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        print("(h) LDA model")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        ldaModel=lda(vectors,id2word)

    # you may complete this part to get answers for part i (similarity in your LDA model)
    if part == "i":
        print("(i): lda-based similarity")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        ldaModel = gensim.models.LdaModel.load('lda_model_v2', mmap='r')
        house = ldaModel[vectors[word2id["house.n"]]]
        home = ldaModel[vectors[word2id["home.n"]]]
        time = ldaModel[vectors[word2id["time.n"]]]
        print("similarity of house and home is {0}".format(cosine_similarity(house, home)))
        print("similarity of house and time is {0}".format(cosine_similarity(house, time)))
        print("similarity of home and time is {0}".format(cosine_similarity(home, time)))

    # you may complete this part to get answers for part j (topic words in your LDA model)
    if part == "j":
        print("(j) get topics from LDA model")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        ldaModel = gensim.models.LdaModel.load('lda_model_complete', mmap='r')
        topics = ldaModel.show_topics(num_topics=10, num_words=10, log=False, formatted=True)
        topicIDs = [topics[j][0] for j in xrange(0,len(topics))]
        for _id in topicIDs:
            realWords = []
            topicWords = get_topic_words(ldaModel,_id)
            for word,probability in topicWords:
                realWords.append(id2word[int(word)])
            print "The words associated with topic {0} are: {1}".format(_id, realWords)
