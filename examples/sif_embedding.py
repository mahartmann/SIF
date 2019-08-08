import sys
import logging
import os
import codecs
from src.data_io import export_embeddings
import argparse

sys.path.append('../src')
from src import data_io, params, SIF_embedding

def read_file(fname):
    with codecs.open(fname, 'r', 'utf-8') as f:
        lines = f.readlines()
    return [line.strip('\n') for line in lines]

def load_sentences(fname):
    lines = read_file(fname)
    idx2sid = dict()
    sents = []
    for i, line in enumerate(lines):
        splt = line.split()
        idx2sid[i] = splt[0]
        sents.append(' '.join(splt[1:]))
    return sents, idx2sid


def setup_logging():
    # create a logger and set parameters
    #logfile = os.path.join(exp_path, logfile)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    #fileHandler = logging.FileHandler(logfile)
    #fileHandler.setFormatter(logFormatter)
    #rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


def main(args):

    # Loading input data
    logging.info('Reading sentences to be converted')
    sentences, idx2sid = load_sentences(args.sentences)
    logging.info('Read {} sentences'.format(len(sentences)))


    # load word vectors
    logging.info('Loading embeddings')
    wordfile = args.word_embs
    We, words, _ = data_io.load_embeddings_from_file(wordfile, dim=args.emb_dim)

    # load word weights
    logging.info('Loading frequencies')
    weightfile = args.freqs
    weightpara = 1e-3  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    rmpc = 1  # number of principal components to remove in SIF weighting scheme
    word2weight = data_io.getWordWeight(weightfile, weightpara)  # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word

    # load sentences
    x, m = data_io.sentences2idx(sentences,
                                 words)  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location

    w = data_io.seq2weight(x, m, weight4ind)  # get word weights

    # set parameters
    sif_params = params.params()
    sif_params.rmpc = rmpc

    # get SIF embedding
    embedding = SIF_embedding.SIF_embedding(We, x, w, sif_params)  # embedding[i,:] is the embedding for sentence i
    export_embeddings(embedding, idx2sid, fname=args.outfile)


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description='Generate SIF sentence embeddings')
    parser.add_argument('--sentences', type=str, help="File with sentences to be converted into embeddings")
    parser.add_argument('--outfile', type=str, help="Name of output file")
    parser.add_argument('--lang', type=str, default='en', help="Language of input data")
    parser.add_argument('--word_embs', type=str, help="File with word embeddings")
    parser.add_argument('--emb_dim', type=int, default=300, help="Dimensionality of word embeddings and resulting sentence embeddings")
    parser.add_argument('--freqs', type=str, help="File with word frequencies")



    args = parser.parse_args()
    main(args)



