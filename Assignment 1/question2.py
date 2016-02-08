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

def load_json():
    lines=open("data/test.txt").readlines()
    # outFile = open("test_n.txt","w")
    i=[]
    sent=[]
    t=[]
    sentid=[]
    for line in lines:
        s = json.loads(line)
        sentid.append(s["id"])
        i.append(s["target_position"])
        sent.append((s["sentence"].split(" ")))
        t.append(s["target_word"])
    return t,i,sent,sentid


'''
(a) function addition for adding 2 vectors
input: vector1
input: vector2
output: addVector, the resulting vector when adding vector1 and vector2
'''
def addition(vector1, vector2):
    if type(vector1[0])!=tuple and type(vector2[0])!=tuple: #and type(vector1[0])!=list and type(vector1[0])!=list:
        vector1 = numpy.asarray(vector1)
        vector2 = numpy.asarray(vector2)
        return vector1+vector2
    v1=vector1
    v2=vector2
    keys1=[]
    keys2=[]
    # vector1=map(int,vector1)
    # vector2=map(int,vector2)
    #convert to dictionary
    dictionary1 = dict((x, y) for x, y in vector1)
    vector1=dictionary1.values() #get all values for the Euclidean distance
    vector1 = map(int, vector1) # provide itemsize for data type
    v1=[] # will be recomuputed
    keys1 = dictionary1.keys() # we need all keys to be able to compare
    #convert to dictionary
    dictionary2 = dict((x, y) for x, y in vector2)
    vector2=dictionary2.values() #get all values for the Euclidean distance
    vector2 = map(int, vector2) # provide itemsize for data type
    v2=[] # will be recomputed
    keys2 = dictionary2.keys() # we need all keys to be able to compare
    result = []
    for key in keys1:
        if key in keys2: # consider only values which indexes appear in both vectors
            result.append((key,(dictionary1[key]+dictionary2[key])))
        else:
            result.append((key,dictionary1[key]))
    for key in keys2:
        if key not in keys1: # consider only values which indexes appear in both vectors
            result.append((key,dictionary2[key]))
    return result

'''
(a) function multiplication for multiplying 2 vectors
input: vector1
input: vector2
output: mulVector, the resulting vector when multiplying vector1 and vector2
'''
def multiplication(vector1, vector2):
    if type(vector1[0])!=tuple and type(vector2[0])!=tuple: #and type(vector1[0])!=list and type(vector1[0])!=list:
        vector1 = numpy.asarray(vector1)
        vector2 = numpy.asarray(vector2)
        return vector1*vector2
    v1=vector1
    v2=vector2
    keys1=[]
    keys2=[]
    #convert to dictionary
    dictionary1 = dict((x, y) for x, y in vector1)
    vector1=dictionary1.values() #get all values for the Euclidean distance
    vector1 = map(int, vector1) # provide itemsize for data type
    v1=[] # will be recomuputed
    keys1 = dictionary1.keys() # we need all keys to be able to compare
    #convert to dictionary
    dictionary2 = dict((x, y) for x, y in vector2)
    vector2=dictionary2.values() #get all values for the Euclidean distance
    vector2 = map(int, vector2) # provide itemsize for data type
    v2=[] # will be recomputed
    keys2 = dictionary2.keys() # we need all keys to be able to compare
    result = []
    for key in keys1:
        if key in keys2: # consider only values which indexes appear in both vectors
            result.append((key,(dictionary1[key]*dictionary2[key]))) 
    return result

'''
(d) function prob_z_given_w to get probability of LDA topic z, given target word w
input: ldaModel
input: topicID as an integer
input: wordVector in frequency space
output: probability of the topic with topicID in the ldaModel, given the wordVector
'''
def prob_z_given_w(ldaModel, topicID, wordVector):
    docGivenTopic=ldaModel[wordVector] # wordVector represents the document
    dictDocGivenTopic = dict((x,y) for x,y in docGivenTopic) 
    wordsGivenTopic = get_topic_words(ldaModel,topicID)
    result = 0.0
    suma = 0.0
    for key in dictDocGivenTopic.keys():
        for word,prob in wordsGivenTopic:
            suma += (dictDocGivenTopic[key]*prob)
    for word,prob in wordsGivenTopic: # !!!!!!!!!! this is wrong
        result += (dictDocGivenTopic[topicID]*prob)/suma

    return result

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
(b) function to set the context documents of a target document t in position i in a 
given sentence sent.
'''
def set_contexts(t,i,sent):
    begin=(int(i)-4)
    end=int(i)+5
    begin=0 if begin<=0 else begin
    end=len(sent) if end>=len(sent) else end
    context_first = [(c,sent[c]) for c in xrange(begin,int(i))]
    context_last = [(c,sent[c]) for c in xrange(int(i)+1, end)]
    cs = context_first+context_last
    
    return cs

def load_tt():
    id2word = {}
    ttdict = {}
    keys = []
    contents=[]
    
    fp = open('data/test_thesaurus.txt')
    _tt = fp.read()
    tt = _tt.split("\n")
    for string in tt:
        _ttdict=string.split("\t")
        keys.append(_ttdict[0])
        contents.append(_ttdict[1].split())

    _dict = zip(keys,contents)
    ttdict=dict(_dict)

    return ttdict

'''
get the best substitution word in a given sentence, according to a given model (tf-idf, word2vec, LDA) and type (addition, multiplication, lda)
'''
def best_substitute(jsonSentence, thesaurus, word2id, model, frequencyVectors, csType):
    t,i,sent = jsonSentence
    contexts=[set_contexts(t[br],i[br],sent[br]) for br in xrange(0,len(jsonSentence[0]))]
    # (b) use addition to get context sensitive vectors
    if csType == "addition":
        br = 0; #indicates the current id of a cycle i.e. count
        finalScore = numpy.zeros(len(contexts)) # list of all final scores unnecessary imo
        finalWord = [[] for y in repeat(None, len(contexts))] #list of returned substitution words
        for context in contexts:
            cs_new = [] #the id of each context word (document) and the list of context words associated with it 
            vt = None
            for position,word in context: # there are four context words before and four after each target word
                try:
                    cs_new.append((word2id[word],model[word])) if type(model)==gensim.models.word2vec.Word2Vec else cs_new.append((word2id[word],model[word2id[word]]))               
                except Exception, e:
                    # print "word {0} does not exist in vectors list".format(word)
                    continue

            try:
                vt = t[br] if type(model)==gensim.models.word2vec.Word2Vec else word2id[t[br]] # the target word in each sentence
            except Exception, e:
                # print "word {0} does not exist in vectors list".format(t[br])
                continue
            # print "cs_new {0}".format(len(cs_new))
            
            if vt is not None:
                allContOfWord = [] # a list of all context words' intersection with target - v(t,C), where c belongs to C
                for contWord in cs_new:
                    add = addition(model[vt],contWord[1]) # must always stay 1
                    allContOfWord.append([add])
                for word in thesaurus[br]: # list of known synonyms
                    score = 0
                    try:
                        vw = model[word] if type(model)==gensim.models.word2vec.Word2Vec else model[word2id[word]] # v(w) from the given equation
                        for vtc in allContOfWord: # sum over every context word in the list of Context words for a given word in a sentence
                            score += cosine_similarity(vw,vtc[0]) # only one vtc anyway
                    except Exception, e:
                        continue
                        # print "no such word {0}".format(word)
                    if score > finalScore[br]:
                        finalScore[br] = score
                        word = word.split(".")
                        finalWord[br] = word[0]
            br = br + 1
        print "went over all contexts"
        return finalWord
         
       
        
    # (c) use multiplication to get context sensitive vectors
    elif csType == "multiplication":
        br = 0; #indicates the current id of a cycle i.e. count
        finalScore = numpy.zeros(len(contexts)) # list of all final scores unnecessary imo
        finalWord = [[] for y in repeat(None, len(contexts))] #list of returned substitution words
        for context in contexts:
            cs_new = [] #the id of each context word (document) and the list of context words associated with it 
            vt = None
            for position,word in context: # there are four context words before and four after each target word
                try:
                    cs_new.append((word2id[word],model[word])) if type(model)==gensim.models.word2vec.Word2Vec else cs_new.append((word2id[word],model[word2id[word]]))          
                except Exception, e:
                    # print "word {0} does not exist in vectors list".format(word)
                    continue
            try:
                vt = t[br] if type(model)==gensim.models.word2vec.Word2Vec else word2id[t[br]] # the target word in each sentence
            except Exception, e:
                # print "word {0} does not exist in vectors list".format(t[br])
                continue
            # print "cs_new {0}".format(len(cs_new))
            
            if vt is not None:
                allContOfWord = [] # a list of all context words' intersection with target - v(t,C), where c belongs to C
                for contWord in cs_new:
                    add = multiplication(model[vt],contWord[1]) # must always stay 1
                    allContOfWord.append([add])
                for word in thesaurus[br]: # list of known synonyms
                    score = 0
                    try:
                        vw = model[word] if type(model)==gensim.models.word2vec.Word2Vec else model[word2id[word]] # v(w) from the given equation
                        for vtc in allContOfWord: # sum over every context word in the list of Context words for a given word in a sentence
                            score += cosine_similarity(vw,vtc[0]) # only one vtc anyway
                    except Exception, e:
                        continue
                        # print "no such word {0}".format(word)
                    if score > finalScore[br]:
                        finalScore[br] = score
                        word = word.split(".")
                        finalWord[br] = word[0]
            br = br + 1

        return finalWord
        
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
            print addition(v3,v4)
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
        id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
        t,i,sent,sentid=load_json()
        thesaurus=load_tt()
        thesaurus = [thesaurus[topic] for topic in t]
        #TF-IDF
        tfIDFVectors = tf_idf(vectors)
        print "Substitute {0}".format(t[0])
        substitute = best_substitute([t,i,sent], thesaurus, word2id, tfIDFVectors, None, "addition")
        outFileTFIDF = open("tf-idf_addition.txt","w")
        for f in xrange(0,len(substitute)):
            outFileTFIDF.write(t[f])
            outFileTFIDF.write(" ")
            outFileTFIDF.write(sentid[f])
            outFileTFIDF.write(" :: ")
            try:
                outFileTFIDF.write(substitute[f])
            except Exception, e:
                outFileTFIDF.write(" ")
            outFileTFIDF.write("\n")
        outFileTFIDF.close
        # word2vec
        w2v=gensim.models.Word2Vec.load('final-model.bin')
        substitute = best_substitute([t,i,sent], thesaurus, word2id, w2v, None, "addition") # what's the point ?
        outFileTFIDF = open("word2vec_addition.txt","w")
        for f in xrange(0,len(substitute)):
            outFileTFIDF.write(t[f])
            outFileTFIDF.write(" ")
            outFileTFIDF.write(sentid[f])
            outFileTFIDF.write(" :: ")
            try:
                outFileTFIDF.write(substitute[f])
            except Exception, e:
                outFileTFIDF.write(" ")
            outFileTFIDF.write("\n")
        outFileTFIDF.close

    # you may complete this to get answers for part c (best substitution words with tf-idf and word2vec, using multiplication)
    if part == "c":
        print("(c) using multiplication to calculate best substitution words")
        id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
        t,i,sent,sentid=load_json()        
        thesaurus=load_tt()
        thesaurus = [thesaurus[topic] for topic in t]
        #TF-IDF
        tfIDFVectors = tf_idf(vectors)
        substitute = best_substitute([t,i,sent], thesaurus, word2id, tfIDFVectors, None, "multiplication")
        outFileTFIDF = open("tf-idf_multiplication.txt","w")
        for f in xrange(0,len(substitute)):
            outFileTFIDF.write(t[f])
            outFileTFIDF.write(" ")
            outFileTFIDF.write(sentid[f])
            outFileTFIDF.write(" :: ")
            try:
                outFileTFIDF.write(substitute[f])
            except Exception, e:
                outFileTFIDF.write(" ")
            outFileTFIDF.write("\n")
        outFileTFIDF.close
        # word2vec
        w2v=gensim.models.Word2Vec.load('final-model.bin')
        substitute = best_substitute([t,i,sent], thesaurus, word2id, w2v, None, "multiplication") # what's the point ?
        outFileTFIDF = open("word2vec_multiplication.txt","w")
        for f in xrange(0,len(substitute)):
            outFileTFIDF.write(t[f])
            outFileTFIDF.write(" ")
            outFileTFIDF.write(sentid[f])
            outFileTFIDF.write(" :: ")
            try:
                outFileTFIDF.write(substitute[f])
            except Exception, e:
                outFileTFIDF.write(" ")
            outFileTFIDF.write("\n")
        outFileTFIDF.close
    
    # this can give you an indication whether your part d1 (P(Z|w) and P(w|Z)) works
    if part == "d":
        print("(d): calculating P(Z|w) and P(w|Z)")
        print("\tloading corpus")
        id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
        print("\tloading LDA model")
        ldaModel = gensim.models.ldamodel.LdaModel.load("lda_model_v2") # change to lda.model !!!
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
