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
    
    def __init__(self, positive_corpus='',negative_corpus=''):
        #caminhos dos diretorios dos corpus
        self.positive_corpus = positive_corpus
        self.negative_corpus = negative_corpus
        
        self.learneddic = {} # {"palavra": {"pos": valor, "neg": valor}}
        # Armazena as probabilidades para cada palavra em: positiva ou negativa
        self.probdic = {}    # {"palavra": {"pos": valor, "neg": valor}} 

    """ Funcoes para treino
        Basicamente, calculam a frequencia das palavras nos arquivos do corpus
    """
    def train_positive(self):
        #freq de palavras nos textos positivos
        self.pos_freq = self.getFrequency(self.positive_corpus)
        #atualiza no dicionario com a categoria
        self.updateLearnedDic(self.pos_freq,"pos")

    def train_negative(self):
        #freq de palavras nos textos negativos
        self.neg_freq = self.getFrequency(self.negative_corpus)
        #atualiza no dicionario com a categoria
        self.updateLearnedDic(self.neg_freq,"neg")
    
    """ Atualiza o dicionario definido em __init__ de acordo com a frequencia da palavra e sua categoria
    """
    def updateLearnedDic(self,wordsfreq,cat):
        for word,freq in wordsfreq.iteritems():
            self.learneddic.setdefault(word,{})
            self.learneddic[word].setdefault(cat,0)
            self.learneddic[word][cat]+=freq
    
    """ Atualiza o dicionario definido em __init__ de acordo com a probabilidade da palavra e sua categoria
    """
    def updateProbDic(self,words,cat):
        for word,freq in wordsfreq.iteritems():
            self.learneddic.setdefault(word,{})
            self.learneddic[word].setdefault(cat,0)
            self.learneddic[word][cat]+=freq
    
    
    """ Calcula a frequencia de palavras para um diretorio de arquivos
    """
    def getFrequency(self,dirc):
        os.chdir(dirc)
        
        #objeto de frequencia inicial (vazio)
        freq = collections.Counter()

        for filename in os.listdir(dirc):
            # leia cada palavra do arquivo
            words = re.findall('\w+', open(filename).read().lower())
            # atualiza a frequencia de palavras no arquivo com a variavel freq
            freq.update(collections.Counter(words))
        return freq
    
    def train2(self, directory, category):
        #objeto de frequencia inicial (vazio)
        freq = collections.Counter()

        for filename in os.listdir(dirc):
            # leia cada palavra do arquivo
            words = re.findall('\w+', open(filename).read().lower())
            # atualiza a frequencia de palavras no arquivo com a variavel freq
            freq.update(collections.Counter(words))
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
        
        # o numero total de palavras de cada categoria
        total_positive = sum(self.pos_freq.values())
        total_negative = sum(self.neg_freq.values())
        
        # Para cada palavra no dicionario de frequencia ...
        for word,freq in self.learneddic.iteritems():
            
            # ajustando entrada para o probdic
            self.probdic.setdefault(word,{})       # adiciona palavra em probdic
            self.probdic[word].setdefault("pos",0)
            self.probdic[word].setdefault("neg",0)
            
            # total de registros positivos e negativos
            positive_count = freq["pos"] if freq.has_key("pos") else 0 #if estranho 
            negative_count = freq["neg"] if freq.has_key("neg") else 0
            
            numerator = (float(positive_count)/total_positive)
            denominator = ((float(positive_count)/total_positive) + (float(negative_count)/total_negative))
            self.probdic[word]["pos"] = numerator/denominator
            
            numerator = (float(negative_count)/total_negative)
            denominator = ((float(negative_count)/total_negative) + (float(positive_count)/total_positive))
            self.probdic[word]["neg"] = numerator/denominator            
   
   ''' Implementar        
    def sum_positive(self):
        """p(S) = (p1 * p2 ... pn) / 
        ( (p1 * p2 ... * pn) + ( (1 - p1) * (1 - p2) ... * (1 - pn) ) )"""
        numerator = sum([token.positive_value for token in self.tokens])
        denominator = numerator + sum([1-token.positive_value for token in self.tokens])
        return float(numerator)/denominator
            
    def sum_negative(self):
        """p(S) = (p1 * p2 ... pn) / 
        ( (p1 * p2 ... * pn) + ( (1 - p1) * (1 - p2) ... * (1 - pn) ) )"""
        numerator = sum([token.negative_value for token in self.tokens])
        denominator = numerator + sum([1-token.negative_value for token in self.tokens])
        return float(numerator)/denominator
    '''
     
