'''
Classificador Naive Bayes em python com NLTK

Baseado em:

- https://github.com/owainlewis/sentiment
- Livro: NLTK, http://nltk.org/book/ch06.html

'''

import nltk
import os
import re


def tokenize(str):
    """ Tokenizer. TODO: definir um tokenizer mais geral
    """
    # ignora pontuacao, que pode ser util em alguns casos,
    # por ex. deteccao de spam
    return re.findall('\w+', str.lower())


class NaiveBayes(object):

    """Classificador Naive Bayes
        positive_corpus =  diretorio com os arquivos do corpus positivo
        positive_corpus =  diretorio com os arquivos do corpus negativo
    """
    def __init__(self, positive_corpus='', negative_corpus=''):

        # caminhos dos diretorios dos corpus
        self.positive_corpus = positive_corpus
        self.negative_corpus = negative_corpus

        # Training set
        self.training_set = []

    ### Funcoes para treinamento

    """ Treinando uma categoria
    """
    def train_category(self, corpus, cat):
        features = self.extract_words(corpus)
        self.training_set = self.training_set + [(self.get_feature(word), cat) for word in features]

    def train(self):
        self.train_category(self.positive_corpus, "pos")
        self.train_category(self.negative_corpus, "neg")
        self.trained_classifier = nltk.NaiveBayesClassifier.train(self.training_set)

    """ Extrair todas as palavras dos arquivos de um diretorio
        As palavras devem aparecer repetidas, o nltk calcula a frequencia automaticamente
    """
    def extract_words(self, dirc):

        words = []
        for filename in os.listdir(dirc):
            with open(os.path.join(dirc, filename)) as f:
                tokens = tokenize(f.read())
                words = words + tokens

        print "Diretorio: ", dirc, " / Arquivos: ", len(os.listdir(dirc))
        return words

    """ Metodo auxiliar, transforma uma palavra em um item de dicionario para o nltk
    """
    def get_feature(self, word):
        return dict([(word, True)])

    """ Metodo auxiliar, transforma uma lista de palavras em uma lista de dicionarios
        para o nltk
    """
    def bag_of_words(self, words):
        return dict([(word, True) for word in words])

    """ Classificando...
        A variavel self.trained_classifier faz o trabalho de classificacao no nltk
        basta passar uma bag of words
    """
    def classifier(self, doc):

        # Lendo documento
        words = []
        with open(doc) as f:
            tokens = tokenize(f.read())
            words = words + tokens

        #Test set
        test_set = self.bag_of_words([w.lower() for w in words])

        return self.trained_classifier.classify(test_set)
