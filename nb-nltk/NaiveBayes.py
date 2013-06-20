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

def extract_words(dirc):
    """ Extrair todas as palavras dos arquivos de um diretorio
        As palavras devem aparecer repetidas, o nltk calcula a frequencia
        automaticamente
    """
    words = []
    for filename in os.listdir(dirc):
        with open(os.path.join(dirc, filename)) as f:
            tokens = tokenize(f.read())
            words = words + tokens

    print "Diretorio: ", dirc, " / Arquivos: ", len(os.listdir(dirc))
    return words

def get_feature(word):
    """ Transforma uma palavra em um item de dicionario para
        o nltk
    """
    return dict([(word, True)])

def bag_of_words(words):
    """ Metodo auxiliar, transforma uma lista de palavras em uma lista de
        dicionarios para o nltk
    """
    return dict([(word, True) for word in words])


class NaiveBayes(object):
    """Classificador Naive Bayes com NLTK mas com classificador simples
        positive_corpus =  diretorio com os arquivos do corpus positivo
        positive_corpus =  diretorio com os arquivos do corpus negativo
    """
    def __init__(self, positive_corpus='', negative_corpus=''):

        # caminhos dos diretorios dos corpus
        self.positive_corpus = positive_corpus
        self.negative_corpus = negative_corpus

        # Training set
        self.training_set = []

    def train_category(self, corpus, cat):
        """ Treinando uma categoria
        """
        features = extract_words(corpus)
        self.training_set = self.training_set + [(get_feature(word), 
            cat) for word in features]

    def train(self):
        self.train_category(self.positive_corpus, "pos")
        self.train_category(self.negative_corpus, "neg")
        self.trained_classifier = nltk.NaiveBayesClassifier.train(
                self.training_set)

    def classifier(self, doc):
        """ Classificando...
            self.trained_classifier faz o trabalho de classificacao no
            nltk basta passar uma bag of words
        """
        # Lendo documento
        words = []
        with open(doc) as f:
            tokens = tokenize(f.read())
            words = words + tokens

        #Test set
        test_set = bag_of_words([w.lower() for w in words])
        return self.trained_classifier.classify(test_set)


class NaiveBayesNltk(object):
    """Classificador Naive Bayes com NLTK mas com classificador usando 
        palavras mais frequentes.
        positive_corpus =  diretorio com os arquivos do corpus positivo
        positive_corpus =  diretorio com os arquivos do corpus negativo
    """
    def __init__(self, positive_corpus='', negative_corpus=''):

        # caminhos dos diretorios dos corpus
        self.positive_corpus = positive_corpus
        self.negative_corpus = negative_corpus

        self.common_words_pos =[]
        self.common_words_neg =[]
        # Training set
        self.training_set = []

    def train_category(self, corpus, cat):
        """ Treinando uma categoria. Escolhe apenas as 20% palavras mais
        frequentes
        """
        features = extract_words(corpus)
    
        if cat == "pos":
            self.common_words_pos = self.common_words_pos + features 
            print "train pos"
        else:
            self.common_words_neg = self.common_words_neg + features
            print 'train neg'

        self.training_set = self.training_set + [(get_feature(word), 
            cat) for word in features]

    def train(self):
        self.train_category(self.positive_corpus, "pos")
        self.train_category(self.negative_corpus, "neg")

        self.trained_classifier = nltk.NaiveBayesClassifier.train(
                self.training_set)

        # palavras mais frequentes de cada cagetoria
        self.common_words_pos = self.get_most_common(self.common_words_pos)
        self.common_words_neg = self.get_most_common(self.common_words_neg)

    def get_most_common(self, words):

        common = nltk.FreqDist(w.lower() for w in words)
        index = int(len(common)*0.50)
        common = common.keys()
        
        return common[:index]


    def classifier(self, doc, cat):
        """ Classificando...
            self.trained_classifier faz o trabalho de classificacao no
            nltk basta passar uma bag of words
        """
        # Lendo documento
        words = []
        with open(doc) as f:
            tokens = tokenize(f.read())
            words = words + tokens

        if cat == "pos":
            most_common = self.common_words_pos
        else:
            most_common = self.common_words_neg    

        bag =[]
        for w in words:
            if w in most_common:
                bag.append(w)
        
        test_set = bag_of_words([w.lower() for w in bag])
        return self.trained_classifier.classify(test_set)
