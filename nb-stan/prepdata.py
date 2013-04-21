#
# prepdata.py
# Prepare data in a corpus for input to Stan
#

import re
import os
import sys


def tokenize(str):
    """ Tokenizer. TODO: definir um tokenizer mais geral
    """
    # ignora pontuacao, que pode ser util em alguns casos,
    # por ex. deteccao de spam
    return re.findall('\w+', str.lower())


class DataConverter(object):

    def __init__(self, pos_corpus, neg_corpus, outfile):
        self.pos_corpus = pos_corpus
        self.neg_corpus = neg_corpus
        self.outfile = outfile

        self.word_count = 0
        self.docno = 0

        # word indices into vocabulary
        self.index = {}

        # word, label and document lists
        self.words = []
        self.labels = []
        self.docs = []

    def update_data(self, corpus_dir, cat_index):
        for fname in os.listdir(corpus_dir):
            self.docno += 1
            self.labels.append(cat_index)
            with open(os.path.join(corpus_dir, fname)) as f:
                tokens = tokenize(f.read())
            for token in tokens:
                if not token in self.index:
                    self.word_count += 1
                    self.index[token] = self.word_count
                self.words.append(self.index[token])
                self.docs.append(self.docno)

    def __write_R_vector(self, f, vec):
        f.write('c(')
        f.write(str(vec[0]) + 'L')
        for el in vec[1:]:
            f.write(', ' + str(el) + 'L')
        f.write(') \n')
        return

    def write_data(self):
        with open(self.outfile, 'w') as f:
            f.writelines(['K <-\n', '2\n',
                          'V <-\n', str(self.word_count) + '\n',
                          'M <-\n', str(self.docno) + '\n',
                          'N <-\n', str(len(self.words)) + '\n', 'z <-\n'])
            self.__write_R_vector(f, self.labels)
            f.writelines(['w <-\n'])
            self.__write_R_vector(f, self.words)
            f.writelines(['doc <-\n'])
            self.__write_R_vector(f, self.docs)
            # alpha hyperparam for 2 classes
            f.writelines(['alpha <-\n', 'c(1, 1)\n'])
            wclassprob = 1.0 / self.word_count
            f.write('beta <-\n c(')
            f.write(str(wclassprob))
            for i in range(1, self.word_count):
                f.write(', ' + str(wclassprob))
            f.write(')\n')

    def convert(self):
        """ Convert data in corpus to dump data format for Stan/JAGS
        """
        print 'Reading positive corpus...'
        self.update_data(self.pos_corpus, 1)
        print 'Reading negative corpus...'
        self.update_data(self.neg_corpus, 2)
        print 'docs: ', self.docno, ' words: ', len(self.words)
        print 'vocab size: ', self.word_count
        self.write_data()
        print 'Data written to ', self.outfile


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: prepdata.py pos_dir neg_dir outfile'
    else:
        converter = DataConverter(sys.argv[1], sys.argv[2], sys.argv[3])
        converter.convert()
