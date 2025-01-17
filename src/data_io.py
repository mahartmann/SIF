from __future__ import print_function

import numpy as np
import pickle
import os
#from tree import tree
#from theano import config

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        if len(i) == 300+1:
            j = 1
            v = []
            while j < len(i):
                v.append(float(i[j]))
                j += 1
            words[i[0]]=n
            We.append(v)
    return (words, np.array(We))


def load_embeddings_from_file(fname, dim, max_vocab=-1):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    with open(fname,  'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and len(line.split()) == 2:
                continue
            else:

                if len(line.rstrip().split(' ', 1)) < 2: continue

                if len(line.rstrip().split(' ')) == dim+1:
                    word, vect = line.rstrip().split(' ', 1)

                    vect = np.fromstring(vect, sep=' ')
                    if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                        vect[0] = 0.01
                    if  word not in word2id:
                        word2id[word] = len(word2id)
                        vectors.append(vect[None])
                # for some ff embeddings of infrequent  word, the  words are mwes
                elif len(line.rstrip().split(' ')) > dim +1:
                    splt = line.rstrip().split(' ')
                    word = '_'.join(splt[:len(splt)-dim])
                    print('Fixed broken embedding for line {}'.format(word))
                    vect = ' '.join(splt[-dim:])

                    vect = np.fromstring(vect, sep=' ')
                    if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                        vect[0] = 0.01
                    if word not in word2id:
                        word2id[word] = len(word2id)
                        vectors.append(vect[None])

            if max_vocab > 0 and i >= max_vocab:
                break

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)
    return embeddings, word2id, id2word,

def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def getSeqs(p1,p2,words):
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    for i in p2:
        X2.append(lookupIDX(words,i))
    return X1, X2

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def getSimEntDataset(f,words,task):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 3:
                if task == "sim":
                    e = (tree(i[0], words), tree(i[1], words), float(i[2]))
                    examples.append(e)
                elif task == "ent":
                    e = (tree(i[0], words), tree(i[1], words), i[2])
                    examples.append(e)
                else:
                    raise ValueError('Params.traintype not set correctly.')

            else:
                print(i)
    return examples

def getSentimentDataset(f,words):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 2:
                e = (tree(i[0], words), i[1])
                examples.append(e)
            else:
                print(i)
    return examples

def getDataSim(batch, nout):
    g1 = []
    g2 = []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x, g1mask = prepare_data(g1)
    g2x, g2mask = prepare_data(g2)

    scores = []
    if nout <=0:
        return (scores, g1x, g1mask, g2x, g2mask)

    for i in batch:
        temp = np.zeros(nout)
        score = float(i[2])
        ceil, fl = int(np.ceil(score)), int(np.floor(score))
        if ceil == fl:
            temp[fl - 1] = 1
        else:
            temp[fl - 1] = ceil - score
            temp[ceil - 1] = score - fl
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype='float32')
    return (scores, g1x, g1mask, g2x, g2mask)

def getDataEntailment(batch):
    g1 = []; g2 = []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x, g1mask = prepare_data(g1)
    g2x, g2mask = prepare_data(g2)

    scores = []
    for i in batch:
        temp = np.zeros(3)
        label = i[2].strip()
        if label == "CONTRADICTION":
            temp[0]=1
        if label == "NEUTRAL":
            temp[1]=1
        if label == "ENTAILMENT":
            temp[2]=1
        scores.append(temp)
    scores = np.matrix(scores)+0.000001
    scores = np.asarray(scores,dtype='float32')
    return (scores,g1x,g1mask,g2x,g2mask)

def getDataSentiment(batch):
    g1 = []
    for i in batch:
        g1.append(i[0].embeddings)

    g1x, g1mask = prepare_data(g1)

    scores = []
    for i in batch:
        temp = np.zeros(2)
        label = i[1].strip()
        if label == "0":
            temp[0]=1
        if label == "1":
            temp[1]=1
        scores.append(temp)
    scores = np.matrix(scores)+0.000001
    scores = np.asarray(scores,dtype='float32')
    return (scores,g1x,g1mask)

def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    oov = set()
    seqs = []
    for sent in sentences:
        seq = []
        toks = sent.split()
        for tok in toks:
            if tok.lower() in words:
                seq.append(words[tok.lower()])
            else:
                oov.add(tok.lower())
        seqs.append(seq)
    print(oov)
    print(len(oov))

    x1,m1 = prepare_data(seqs)
    return x1, m1


def sentiment2idx(sentiment_file, words):
    """
    Read sentiment data file, output array of word indices that can be fed into the algorithms.
    :param sentiment_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, golds. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location), golds[i] is the label (0 or 1) for sentence i.
    """
    f = open(sentiment_file,'r')
    lines = f.readlines()
    golds = []
    seq1 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; score = int(i[1]) # score are labels 0 and 1
        X1 = getSeq(p1,words)
        seq1.append(X1)
        golds.append(score)
    x1,m1 = prepare_data(seq1)
    return x1, m1, golds

def sim2idx(sim_file, words):
    """
    Read similarity data file, output array of word indices that can be fed into the algorithms.
    :param sim_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, x2, m2, golds. x1[i, :] is the word indices in the first sentence in pair i, m1[i,:] is the mask for the first sentence in pair i (0 means no word at the location), golds[i] is the score for pair i (float). x2 and m2 are similar to x1 and m2 but for the second sentence in the pair.
    """
    f = open(sim_file,'r')
    lines = f.readlines()
    golds = []
    seq1 = []
    seq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        X1, X2 = getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1,m1 = prepare_data(seq1)
    x2,m2 = prepare_data(seq2)
    return x1, m1, x2, m2, golds

def entailment2idx(sim_file, words):
    """
    Read similarity data file, output array of word indices that can be fed into the algorithms.
    :param sim_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, x2, m2, golds. x1[i, :] is the word indices in the first sentence in pair i, m1[i,:] is the mask for the first sentence in pair i (0 means no word at the location), golds[i] is the label for pair i (CONTRADICTION NEUTRAL ENTAILMENT). x2 and m2 are similar to x1 and m2 but for the second sentence in the pair.
    """
    f = open(sim_file,'r')
    lines = f.readlines()
    golds = []
    seq1 = []
    seq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = i[2]
        X1, X2 = getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1,m1 = prepare_data(seq1)
    x2,m2 = prepare_data(seq2)
    return x1, m1, x2, m2, golds

def getWordWeight(weightfile, a=1e-3):
    if a <=0: # when the parameter makes no sense, use unweighted
        a = 1.0

    word2weight = {}
    with open(weightfile) as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            if(len(i) == 2):
                word2weight[i[1]] = float(i[0])
                N += float(i[0])
            else:
                print(i)
    for key, value in word2weight.items():
        word2weight[key] = a / (a + value/N)
    return word2weight

def getWeight(words, word2weight):
    weight4ind = {}
    for word, ind in words.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind

def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i,j] > 0 and seq[i,j] >= 0:
                weight[i,j] = weight4ind[seq[i,j]]
    weight = np.asarray(weight, dtype='float32')
    return weight


def export_embeddings(embs, idx2sid, fname):
    with open(fname, 'w', encoding='utf-8') as f:
        for i in range(embs.shape[0]):
            f.write("{} {}\n".format(idx2sid[i], " ".join('%.5f' % x for x in embs[i, :])))
    f.close()


def export_embeddings_for_browser(embs, idx2sid, fname):
    # write embeddings
    with open(fname + '.embs', 'w', encoding='utf-8') as f:
        for i in range(embs.shape[0]):
            f.write("{}\n".format("\t".join('%.5f' % x for x in embs[i, :])))
    f.close()
    with open(fname + '.meta', 'w', encoding='utf-8') as f:
        f.write('Sentence_id\n')
        for i in range(embs.shape[0]):
            f.write("{}\n".format(idx2sid[i]))
    f.close()


    # write meta data

def getIDFWeight(wordfile, save_file=''):
    def getDataFromFile(f, words):
        f = open(f,'r')
        lines = f.readlines()
        golds = []
        seq1 = []
        seq2 = []
        for i in lines:
            i = i.split("\t")
            p1 = i[0]; p2 = i[1]; score = float(i[2])
            X1, X2 = getSeqs(p1,p2,words)
            seq1.append(X1)
            seq2.append(X2)
            golds.append(score)
        x1,m1 = prepare_data(seq1)
        x2,m2 = prepare_data(seq2)
        return x1,m1,x2,m2



    prefix = "../data/"
    farr = ["MSRpar2012"]
    #farr = ["MSRpar2012",
    #        "MSRvid2012",
    #        "OnWN2012",
    #        "SMTeuro2012",
    #        "SMTnews2012", # 4
    #        "FNWN2013",
    #        "OnWN2013",
    #        "SMT2013",
    #        "headline2013", # 8
    #        "OnWN2014",
    #        "deft-forum2014",
    #        "deft-news2014",
    #        "headline2014",
    #        "images2014",
    #        "tweet-news2014", # 14
    #        "answer-forum2015",
    #        "answer-student2015",
    #        "belief2015",
    #        "headline2015",
    #        "images2015",    # 19
    #        "sicktest",
    #        "twitter",
    #        "JHUppdb",
    #        "anno-dev",
    #        "anno-test"]
    (words, We) = getWordmap(wordfile)
    df = np.zeros((len(words),))
    dlen = 0
    for f in farr:
        g1x,g1mask,g2x,g2mask = getDataFromFile(prefix+f, words)
        dlen += g1x.shape[0]
        dlen += g2x.shape[0]
        for i in range(g1x.shape[0]):
            for j in range(g1x.shape[1]):
                if g1mask[i,j] > 0:
                    df[g1x[i,j]] += 1
        for i in range(g2x.shape[0]):
            for j in range(g2x.shape[1]):
                if g2mask[i,j] > 0:
                    df[g2x[i,j]] += 1

    weight4ind = {}
    for i in range(len(df)):
        weight4ind[i] = np.log2((dlen+2.0)/(1.0+df[i]))
    if save_file:
        pickle.dump(weight4ind, open(save_file, 'w'))
    return weight4ind
