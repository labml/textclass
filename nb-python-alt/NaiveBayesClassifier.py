'''
Classificador Naive Bayes em python.
- Cria um dicionario para a classificacao da frequencia das palavras
- Cria um segundo Dicionario com as probabilidades baseado na frequencia das palavras
- O corpus utilizado: classificador de reviews de filmes do NLTK

Baseado em:

- http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
- Livro: Programming Collective Intelligence (cap. Document Filtering)

'''
import os, re, collections, math

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
        Os passos para o treinamento:
            1. calcular a frequencia (numero de vezes que a palavra aparece nos documentos...)
            2. armazenar a quantidade de documentos que foram usados para essa categoria
            3. armazenar as frequencias da palavra em um dicionario de acordo com sua categoria

            Resultado final: {"palavra": 
                                    {"pos":111,"neg":222}
                             }

    """
    def train_positive(self):
        #freq de palavras nos textos positivos
        self.pos_freq = self.getFrequency(self.positive_corpus)
        
        # quantos documentos nessa categoria?
        self.count_positive = len(os.listdir(self.positive_corpus))
        
        #atualiza no dicionario com a categoria
        self.updateLearnedDic(self.pos_freq,"pos")

    def train_negative(self):
        #freq de palavras nos textos negativos
        self.neg_freq = self.getFrequency(self.negative_corpus)
        
        # quantos documentos nessa categoria?
        self.count_negative = len(os.listdir(self.negative_corpus))
        
        #atualiza no dicionario com a categoria
        self.updateLearnedDic(self.neg_freq,"neg")
    
    """ Atualiza o dicionario definido em __init__ de acordo com a frequencia da palavra e sua categoria
    """
    def updateLearnedDic(self,wordsfreq,cat):
        for word,freq in wordsfreq.iteritems():
            # entrada default
            self.learneddic.setdefault(word,{})
            self.learneddic[word].setdefault("pos",0)
            self.learneddic[word].setdefault("neg",0)
            # atualiza
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
            
        print "Diretorio: ", dirc, " / Arquivos: ", len(os.listdir(dirc)), " / Palavras unicas: ", len(freq)
        return freq

    """ Operacoes com Naive Bayes
    """
    def calculate_probabilities(self):
        """ Faz o calculo de probabilidades para determinar o quanto cada palavra pode ser positiva ou negativa
        """
        weight=1.0
        ap=0.5
        
        for word,freq in self.learneddic.iteritems():
            fprob_pos = float(freq["pos"]) / float(self.count_positive)
            fprob_neg = float(freq["neg"]) / float(self.count_negative)
            
            sum_word = freq["pos"] + freq["neg"]  
            
            # ajustando entrada para o dicionario de probabilidades
            self.probdic.setdefault(word,{})       
            self.probdic[word].setdefault("pos",0)
            self.probdic[word].setdefault("neg",0)
            
            # formula = ((weight*ap)+(totals*basicprob))/ (weight+totals)
            self.probdic[word]["pos"] = ((weight * ap) + (sum_word + fprob_pos)) / (weight + sum_word)
            self.probdic[word]["neg"] = ((weight * ap) + (sum_word + fprob_neg)) / (weight + sum_word)
            
            
    """ Operacoes de classificacao
    """
    def naive_bayes(self, doc):
        # extrair todas as palavras do documento
        doc_words = re.findall('\w+', open(doc).read().lower())
        doc_words = collections.Counter(doc_words)
        
        # para cada palavra do documento a ser classificado
        # procura no dicionario de probabilidades e salva...
        doc_prob = {}
        for word, prob in self.probdic.iteritems():
            if doc_words.has_key(word):
                doc_prob.setdefault(word,prob)
            
        """Calcular Pr(Documento|Categoria):
            - para categoria:
                - multiplicar as probabilidades de cada palavra que o documento a ser classificado tem em comum no dicionario de
                  probabilidades
        """ 
        pmult_pos = 0
        pmult_neg = 0
        for word,prob in doc_prob.iteritems():
            # trocando multiplicacao de probabilidades por soma de logaritmos
            pmult_pos += math.log(prob["pos"])
            pmult_neg += math.log(prob["neg"])

        """Probabilidade Bayesiana agora... 
            Queremos calcular Pr(Categoria | Documento) = Pr(Documento | Categoria) x Pr(Categoria) / Pr(Documento)

            Como Pr(Documento) e  um valor comum para todas as categorias, ele e eliminado pois nao afeta o resultado final 
            Pr(Categoria) = numero de documentos na categoria / total de documentos

            Pr(Categoria|Documento) = log Pr(Documento | Categoria) + log Pr(Categoria) 
        """
        total_doc = self.count_positive + self.count_negative
        
        # probabilidade de que o documento seja positivo...
        prob_doc_pos = math.log(float(self.count_positive) / float(total_doc)) + pmult_pos
       
        # probabilidade de que o documento seja negativo...
        prob_doc_neg = math.log(float(self.count_negative) / float(total_doc)) + pmult_neg 
   
        #resultado...
        if prob_doc_pos > prob_doc_neg: 
            return "POSITIVO"
        else:
            return "NEGATIVO"
