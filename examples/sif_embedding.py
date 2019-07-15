import sys
import logging

sys.path.append('../src')
import data_io, params, SIF_embedding, codecs

def read_file(fname):
    with codecs.open(fname, 'r', 'utf-8') as f:
        lines = f.readlines()
    return [line.strip('\n') for line in lines]

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

logging.info('Readind sentences')
sentences = read_file('/home/lwp876/ira/data/ira_tweets_csv_hashed.csv.ru.tokenized.text.1000')
logging.info('Read {} sentences'.format(len(sentences)))
# input
logging.info('Loading embeddings')
wordfile = '/home/lwp876/resources/wiki.ru.align.vec' # word vector file, can be downloaded from GloVe website
logging.info('Loading s')
weightfile = '../auxiliary_data/ira_ru_freqs.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
#sentences = ['this is an example sentence', 'this is another sentence that is slightly longer']

# load word vectors
(words, We) = data_io.getWordmap(wordfile)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# load sentences
x, m, _ = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
w = data_io.seq2weight(x, m, weight4ind) # get word weights

# set parameters
params = params.params()
params.rmpc = rmpc
# get SIF embedding
embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i
print(embedding.shape)