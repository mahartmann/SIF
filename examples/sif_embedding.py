import sys
import logging
import os
import codecs
from src.data_io import export_embeddings

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


def setup_logging(exp_path='.', logfile='log.txt'):
    # create a logger and set parameters
    logfile = os.path.join(exp_path, logfile)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

logging.info('Reading sentences')
sentences, idx2sid= read_file('/home/lwp876/ira/data/ira_tweets_csv_hashed.csv.ru.tokenized.text.100000')
#sentences, idx2sid = load_sentences('/home/mareike/PycharmProjects/sheffield/data/ira_tweets_csv_hashed.csv.ru.tokenized.10000')
logging.info('Read {} sentences'.format(len(sentences)))
# input
logging.info('Loading embeddings')
wordfile = '/home/lwp876/ira/resources/wiki.ru.align.vec' # word vector file, can be downloaded from GloVe website
#wordfile = '/home/mareike/PycharmProjects/sheffield/data/wiki.ru.align.vec' # word vector file, can be downloaded from GloVe website
logging.info('Loading frequencies')
weightfile = '../auxiliary_data/ira_ru_freqs.txt' # each line is a word and its frequency
#weightfile = '/home/mareike/PycharmProjects/sheffield/data/ira_tweets_csv_hashed.csv.ru.tokenized.freqs'
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
#sentences = ['this is an example sentence', 'this is another sentence that is slightly longer']

# load word vectors
We, words, _ = data_io.load_embeddings_from_file(wordfile, dim=300)
print(len(words))
print(We)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# load sentences
x, m  = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
print(x)
print(m)
w = data_io.seq2weight(x, m, weight4ind) # get word weights

# set parameters
params = params.params()
params.rmpc = rmpc
# get SIF embedding
embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i
print(embedding.shape)

export_embeddings(embedding, idx2sid, '.', lang = 'ru', emb_dim = 300)