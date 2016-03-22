if __name__ == "__main__":
    import sys
    from utils import *
    from rnnmath import *
    from rnn import *
    mode = sys.argv[1].lower()
    
    if mode == "train":
        '''
        starter code for parameter estimation.
        change this to different values, or use it to get you started with your own testing class
        '''
        data_folder = sys.argv[2]
        vocabsize = 2000
        # hdim = 5

        curMinLoss = 999999999999999
        bestHid = 0
        bestLR = 0
        bestBP = 0
        # get the data set vocabulary
        vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
        num_to_word = dict(enumerate(vocab.index[:vocabsize]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocabsize
        fraction_lost = fraq_loss(vocab, word_to_num, vocabsize)
        print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocabsize, len(vocab), 100*(1-fraction_lost)))

        docs = load_dataset(data_folder + '/ptb-train.txt')
        S_train = docs_to_indices(docs, word_to_num)
        X_train, D_train = seqs_to_lmXY(S_train)

        # Load the dev set (for tuning hyperparameters)
        docs = load_dataset(data_folder + '/ptb-dev.txt')
        S_dev = docs_to_indices(docs, word_to_num)
        X_dev, D_dev = seqs_to_lmXY(S_dev)

        X = X_train[:1000]
        D = D_train[:1000]
        X_dev = X_dev[:1000]
        D_dev = D_dev[:1000]
        hidun = array([10,25,50,100])
        backprop = array([0,3,10])
        learnrate = array([0.5,0.1,0.005])
        annealing = array([0,3,5])
        for dim in range(0,4):
                r = RNN(vocabsize, hidun[dim])
                # train the RNN
                #r.train(X, D, X_dev, D_dev, epochs, learning_rate, anneal, back_steps, batch_size, min_change):
                for bp in range(0,3):
                        for lr in range(0,3):
                            for an in range(0,3):
                                print "--------------------HIDDEN UNITS-------------------------"
                                print hidun[dim]
                                print "---------------------------------------------------------"
                                print "--------------------BACKPROP Steps-------------------------"
                                print backprop[bp]
                                print "---------------------------------------------------------"
                                print "--------------------LEARNING RATE-------------------------"
                                print learnrate[lr]
                                print "---------------------------------------------------------"
                                print "--------------------Annealing-------------------------"
                                print annealing[an]
                                print "---------------------------------------------------------"
                                r.train(X, D, X_dev, D_dev, 10, learnrate[lr], annealing[an], backprop[bp], 100, 0.0001)

                                dev_loss = r.compute_mean_loss(X_dev, D_dev)
                                print("mean loss: {0}".format(dev_loss))

                                # q = best unigram frequency from omitted vocab
                                # this is the best expected loss out of that set
                                q = vocab.freq[vocabsize] / sum(vocab.freq[vocabsize:])
                                print "Unadjusted: %.03f" % exp(dev_loss)
                                print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost, q))
                                print "----------------------END----------------------------------"
                                # save RNN matricse
                                if curMinLoss > dev_loss:
                                    curU = r.U
                                    curV = r.V
                                    curW = r.W
                                    save("rnn-0.U", r.U)
                                    save("rnn-0.V", r.V)
                                    save("rnn-0.W", r.W)
                                    curMinLoss = dev_loss
                                    bestHid = hidun[dim]
                                    bestLR = backprop[bp]
                                    bestBP = learnrate[lr]

        save("rnn.U", curU)
        save("rnn.V", curV)
        save("rnn.W", curW)

        print "VALUES:"
        print bestHid
        print bestLR
        print bestBP