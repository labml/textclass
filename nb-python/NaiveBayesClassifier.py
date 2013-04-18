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
        self.counts = {}

        # {"palavra": {"pos": probabilidade, "neg": probabilidade}}
        self.probs = {}

        self.pos_freq = {}
        self.neg_freq = {}

        # TODO: definir conjunto de classes como um campo do objeto,
        # p/ generalizar classificacao

        # variavel para coleta de informacoes estatisticas
        self.info = ""

    def train(self):
        pass

    def train_class():
        """
        """
        pass

    """ Funcoes para treino
        Basicamente, calculam a frequencia das palavras nos arquivos do corpus
    """
    def train_positive(self):
        # freq de palavras nos textos positivos
        self.pos_freq = self.getFrequency(self.positive_corpus)
        # atualiza no dicionario com a categoria
        self.update_counts(self.pos_freq, "pos")

    def train_negative(self):
        # freq de palavras nos textos negativos
        self.neg_freq = self.getFrequency(self.negative_corpus)
        # atualiza no dicionario com a categoria
        self.update_counts(self.neg_freq, "neg")

    """ Atualiza o dicionario definido em __init__ de acordo com a frequencia da palavra e sua categoria
    """
    def update_counts(self, wordsfreq, cat):
        for word, freq in wordsfreq.iteritems():
            self.counts.setdefault(word, {})
            self.counts[word].setdefault(cat, 0)
            self.counts[word][cat] += freq

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
            p(c=1 | w) =          p(w | c=1)p(c=1)
                        -------------------------------------
                         p(w | c=1)p(c=1)  + p(w | c=2)p(c=2)

            Como o fator normalizador e' o mesmo para as duas classes,
            e so' queremos achar a maior probabilidade, o denominador nao
            precisa ser calculado.

            p(w | c=1) = (n. de w em itens positivos / n. de palavras nos itens positivos)

            Laplace smoothing adiciona contagens ficticias no numerador e
            denominador para suavizar as probabilidades de palavras nao vistas em
            alguma classe
        """

        # o numero total de palavras de cada categoria
        total_positive = sum(self.pos_freq.values())
        total_negative = sum(self.neg_freq.values())

        vocab_size = len(self.counts.keys())

        # TODO: calculate priors for classes

        # Para cada palavra no dicionario de frequencia ...
        for word, freq in self.counts.iteritems():

            # ajustando entrada para o dicionario de probabilidades
            self.probs.setdefault(word, {})       # adiciona palavra em probs
            self.probs[word].setdefault("pos", 0)
            self.probs[word].setdefault("neg", 0)

            # total de registros positivos e negativos (+1 for Laplace smoothing)
            positive_count = freq.get("pos", 0) + 1
            negative_count = freq.get("neg", 0) + 1

            pw_given_pos = (float(positive_count) / (total_positive + vocab_size))
            pw_given_neg = (float(negative_count) / (total_negative + vocab_size))

            self.probs[word]["pos"] = pw_given_pos
            self.probs[word]["neg"] = pw_given_neg

    """ Operacoes de classificacao
    """
    def classifier(self, dir_positivo, dir_negativo):
        # Para cada diretorio, percorrer os arquivos de texto e contar a frequencia geral das palavras
        freq_geral = self.getFrequency(dir_positivo)
        freq_geral.update(self.getFrequency(dir_negativo))

        prob_geral = {}
        # Para cada palavra a ser classificada, procurar no dicionario de probabilidades treinado
        for word, prob in self.probs.iteritems():
            if word in freq_geral:
                prob_geral.setdefault(word, prob)
        return prob_geral

    """ Classifica um item do conjunto de teste
    """
    def classify_item(self, item):
        words = tokenize(item)

        # ignora palavras que nao estao no dicionario de treinamento (probs)
        # TODO: se a palavra nao existe em probs, usar prob. de "palavra desconhecida"
        pos_probs = [self.probs[w]['pos'] for w in words if w in self.probs]
        neg_probs = [self.probs[w]['neg'] for w in words if w in self.probs]
        total_pos_logprob = sum([math.log(p) for p in pos_probs if p > 0.0])
        total_neg_logprob = sum([math.log(p) for p in neg_probs if p > 0.0])
        item_class = 'pos'
        if total_neg_logprob > total_pos_logprob:
            item_class = 'neg'
        return item_class
