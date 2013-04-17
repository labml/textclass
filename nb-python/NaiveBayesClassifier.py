import os
import re
import math
import collections

""" Tokenizer. TODO: definir um tokenizer mais geral
"""
def tokenize(str):
    # ignora pontuacao, que pode ser util em alguns casos,
    # por ex. deteccao de spam
    return re.findall('\w+', str.lower())

'''
Classificador Naive Bayes em python.
- Cria um dicionario para a classificacao da frequencia das palavras
- Cria um segundo Dicionario com as probabilidades baseado na frequencia das palavras
- O corpus utilizado: classificador de reviews de filmes do NLTK

Baseado no codigo disponivel em:

- https://github.com/stuntgoat/Generic-Naive-Bayesian-Classifier
- https://github.com/euclid1729/naive-bayes-spam-classifier-

TODO: Implementar classificacao entre textos
'''
class NaiveBayesClassifier(object):
    """Classificador Naive Bayes
        positive_corpus =  diretorio com os arquivos do corpus positivo
        positive_corpus =  diretorio com os arquivos do corpus negativo
    """

    def __init__(self, positive_corpus='', negative_corpus=''):
        # caminhos dos diretorios dos corpus
        self.positive_corpus = positive_corpus
        self.negative_corpus = negative_corpus

        # {"palavra": {"pos": n_ocorrencias, "neg": n_ocorrencias}}
        self.learneddic = {}

        # {"palavra": {"pos": probabilidade, "neg": probabilidade}}
        self.probdic = {}

        self.pos_freq = {}
        self.neg_freq = {}

        # TODO: definir conjunto de classes como um campo do objeto,
        # p/ generalizar classificacao

        # variavel para coleta de informacoes estatisticas
        self.info = ""

    """ Funcoes para treino
        Basicamente, calculam a frequencia das palavras nos arquivos do corpus
    """
    def train_positive(self):
        # freq de palavras nos textos positivos
        self.pos_freq = self.getFrequency(self.positive_corpus)
        # atualiza no dicionario com a categoria
        self.updateLearnedDic(self.pos_freq, "pos")

    def train_negative(self):
        # freq de palavras nos textos negativos
        self.neg_freq = self.getFrequency(self.negative_corpus)
        # atualiza no dicionario com a categoria
        self.updateLearnedDic(self.neg_freq, "neg")

    """ Atualiza o dicionario definido em __init__ de acordo com a frequencia da palavra e sua categoria
    """
    def updateLearnedDic(self, wordsfreq, cat):
        for word, freq in wordsfreq.iteritems():
            self.learneddic.setdefault(word, {})
            self.learneddic[word].setdefault(cat, 0)
            self.learneddic[word][cat] += freq

    """ Calcula a frequencia de palavras para um diretorio de arquivos
    """
    def getFrequency(self, dirc):
        # objeto de frequencia inicial (vazio)
        freq = collections.Counter()

        for filename in os.listdir(dirc):
            # le conteudo do arquivo e atualiza
            # frequencia de ocorrencia de cada palavra
            with open(os.path.join(dirc, filename)) as f:
                words = tokenize(f.read())
                freq.update(words)

        print "Diretorio: ", dirc, " / Arquivos: ", len(os.listdir(dirc)), " / Palavras unicas: ", len(freq)
        return freq

    """ Operacoes com Naive Bayes
    """
    def calculate_probabilities(self):
        """ Faz o calculo de probabilidades para determinar o quanto cada
            palavra pode ser positiva ou negativa.

            Para cada palavra:
            prob. pos. = (valor pos. / total de palavras pos)
                             /
                   (valor pos. / total de palavras pos) + (valor neg. / total de palavras neg.)
        """
        # TODO: implementar Laplace smoothing

        # o numero total de palavras de cada categoria
        total_positive = sum(self.pos_freq.values())
        total_negative = sum(self.neg_freq.values())

        # Para cada palavra no dicionario de frequencia ...
        for word, freq in self.learneddic.iteritems():

            # ajustando entrada para o probdic
            self.probdic.setdefault(word, {})       # adiciona palavra em probdic
            self.probdic[word].setdefault("pos", 0)
            self.probdic[word].setdefault("neg", 0)

            # total de registros positivos e negativos
            positive_count = freq.get("pos", 0)
            negative_count = freq.get("neg", 0)

            numerator = (float(positive_count) / total_positive)
            denominator = ((float(positive_count) / total_positive) +
                           (float(negative_count) / total_negative))
            self.probdic[word]["pos"] = numerator / denominator

            numerator = (float(negative_count) / total_negative)
            denominator = ((float(negative_count) / total_negative) +
                           (float(positive_count) / total_positive))
            self.probdic[word]["neg"] = numerator / denominator

    """ Operacoes de classificacao
    """
    def classifier(self, dir_positivo, dir_negativo):
        # Para cada diretorio, percorrer os arquivos de texto e contar a frequencia geral das palavras
        freq_geral = self.getFrequency(dir_positivo)
        freq_geral.update(self.getFrequency(dir_negativo))

        prob_geral = {}
        # Para cada palavra a ser classificada, procurar no dicionario de probabilidades treinado
        for word, prob in self.probdic.iteritems():
            if word in freq_geral:
                prob_geral.setdefault(word, prob)
        return prob_geral

    """ Classifica um item do conjunto de teste
    """
    def classify_item(self, item):
        words = tokenize(item)

        # ignora palavras que nao estao no dicionario de treinamento (probdic)
        # TODO: se a palavra nao existe em probdic, usar prob. de "palavra desconhecida"
        pos_probs = [self.probdic[w]['pos'] for w in words if w in self.probdic]
        neg_probs = [self.probdic[w]['neg'] for w in words if w in self.probdic]
        total_pos_logprob = sum([math.log(p) for p in pos_probs if p > 0.0])
        total_neg_logprob = sum([math.log(p) for p in neg_probs if p > 0.0])
        item_class = 'pos'
        if total_neg_logprob > total_pos_logprob:
            item_class = 'neg'
        return item_class

    def sum_positive(self, probdic_teste):
        """p(S) = (p1 * p2 ... pn)
                    /
                  ( (p1 * p2 ... * pn) + ( (1 - p1) * (1 - p2) ... * (1 - pn) ) )
        """
        # nao muito eficiente, refazer depois de testar
        numerator = sum([prob["pos"] for word, prob in probdic_teste.iteritems()])
        denominator = numerator + sum([1 - prob["pos"] for word, prob in probdic_teste.iteritems()])
        return float(numerator) / denominator

    def sum_negative(self, probdic_teste):
        """p(S) = (p1 * p2 ... pn)
                    /
             ( (p1 * p2 ... * pn) + ( (1 - p1) * (1 - p2) ... * (1 - pn) ) )"""

        numerator = sum([prob["neg"] for word, prob in probdic_teste.iteritems()])
        denominator = numerator + sum([1 - prob["neg"] for word, prob in probdic_teste.iteritems()])
        return float(numerator) / denominator
