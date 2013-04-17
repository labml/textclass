import re
import collections

""" Tokenizer. TODO: definir um tokenizer mais geral
"""
def tokenize(str):
    # ignora pontuacao, que pode ser util em alguns casos,
    # por ex. deteccao de spam
    return re.findall('\w+', str)

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

        self.learneddic = {} # {"palavra": {"pos": n_ocorrencias, "neg": n_ocorrencias}}
        # Armazena as probabilidades para cada palavra em: positiva ou negativa
        self.probdic = {}    # {"palavra": {"pos": probabilidade, "neg": probabilidade}}

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

    """ Atualiza o dicionario definido em __init__ de acordo com a probabilidade da palavra e sua
    categoria
    """
    def updateProbDic(self, words, cat):
        for word, freq in wordsfreq.iteritems():
            self.learneddic.setdefault(word, {})
            self.learneddic[word].setdefault(cat, 0)
            self.learneddic[word][cat] += freq


    """ Calcula a frequencia de palavras para um diretorio de arquivos
    """
    def getFrequency(self, dirc):
        os.chdir(dirc)

        #objeto de frequencia inicial (vazio)
        freq = collections.Counter()

        for filename in os.listdir(dirc):
            # leia cada palavra do arquivo
            words = tokenize(open(filename).read().lower())
            # atualiza a frequencia de palavras no arquivo com a variavel freq
            freq.update(collections.Counter(words))

        print "Diretorio: ", dirc, " / Arquivos: ", len(os.listdir(dirc)), " / Palavras únicas: ", len(freq)
        return freq

    """ Operacoes com Naive Bayes
    """
    def calculate_probabilities(self):
        """ Faz o calculo de probabilidades para determinar o quanto cada palavra pode ser positiva ou negativa

            Para cada palavra:
            prob. pos. = (valor pos. / total de palavras pos)
                             /
                         (valor pos. / total de palavras pos) + (valor neg. / total de palavras neg.)
        """
        # TODO implementar Laplace smoothing

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
            positive_count = freq["pos"] if freq.has_key("pos") else 0 #if estranho
            negative_count = freq["neg"] if freq.has_key("neg") else 0

            numerator = (float(positive_count)/total_positive)
            denominator = ((float(positive_count)/total_positive) + (float(negative_count)/total_negative))
            self.probdic[word]["pos"] = numerator/denominator

            numerator = (float(negative_count)/total_negative)
            denominator = ((float(negative_count)/total_negative) + (float(positive_count)/total_positive))
            self.probdic[word]["neg"] = numerator/denominator


    """ Operacoes de classificacao
    """
    def classifier(self, dir_positivo, dir_negativo):
        #Para cada diretorio, percorrer os arquivos de texto e contar a frequencia geral das palavras
        freq_geral = self.getFrequency(dir_positivo)
        freq_geral.update(self.getFrequency(dir_negativo))

        prob_geral = {}
        #Para cada palavra a ser classificada, procurar no dicionario de probabilidades treinado
        for word, prob in self.probdic.iteritems():
            if freq_geral.has_key(word):
                prob_geral.setdefault(word, prob)
        return prob_geral

    """ Classifica um item do conjunto de teste
    """
    def classify_item(self, item):
        words = tokenize(item)
        # TODO: se a palavra nao existe em probdic, usar prob. de "palavra desconhecida"
        pos_probs = [self.probdic[word]['pos'] for word in words]
        return 0 # TODO: implementar o resto

    def sum_positive(self, probdic_teste):
        """p(S) = (p1 * p2 ... pn)
                    /
                  ( (p1 * p2 ... * pn) + ( (1 - p1) * (1 - p2) ... * (1 - pn) ) )
        """
        # nao muito eficiente, refazer depois de testar
        numerator = sum([prob["pos"] for word, prob in probdic_teste.iteritems()])
        denominator = numerator + sum([1-prob["pos"] for word, prob in probdic_teste.iteritems()])
        return float(numerator)/denominator

    def sum_negative(self, probdic_teste):
        """p(S) = (p1 * p2 ... pn)
                    /
                  ( (p1 * p2 ... * pn) + ( (1 - p1) * (1 - p2) ... * (1 - pn) ) )"""

        numerator = sum([prob["neg"] for word, prob in probdic_teste.iteritems()])
        denominator = numerator + sum([1-prob["neg"] for word, prob in probdic_teste.iteritems()])
        return float(numerator)/denominator
