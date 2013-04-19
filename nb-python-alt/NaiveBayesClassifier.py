'''
Classificador Naive Bayes em python.
- Cria um dicionario para a classificacao da frequencia das palavras
- Cria um segundo Dicionario com as probabilidades baseado na frequencia das palavras
- O corpus utilizado: classificador de reviews de filmes do NLTK

Baseado em:

- http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
- Livro: Programming Collective Intelligence (cap. Document Filtering)

'''
import os
import re
import collections
import math

def tokenize(str):
    """ Tokenizer. TODO: definir um tokenizer mais geral
    """
    # ignora pontuacao, que pode ser util em alguns casos,
    # por ex. deteccao de spam
    return re.findall('\w+', str.lower())


class NaiveBayesClassifier(object):
    """Classificador Naive Bayes
        positive_corpus =  diretorio com os arquivos do corpus positivo
        positive_corpus =  diretorio com os arquivos do corpus negativo
    """
    
    def __init__(self, positive_corpus='',negative_corpus=''):
        # caminhos dos diretorios dos corpus
        self.positive_corpus = positive_corpus
        self.negative_corpus = negative_corpus

        # {"pos": n_documentos_positivos, "neg": n_documentos_negativos}
        self.doc_counts = {} 
        
        # {"palavra": {"pos": n_ocorrencias, "neg": n_ocorrencias}}
        self.counts = {}

        # {"palavra": {"pos": probabilidade, "neg": probabilidade}}
        self.probs = {}

    """ Funcoes para treino
        Os passos para o treinamento:
            1. calcular a frequencia (numero de vezes que a palavra aparece nos documentos...)
            2. armazenar a quantidade de documentos que foram usados para essa categoria
            3. armazenar as frequencias da palavra em um dicionario de acordo com sua categoria

            Resultado final: {"palavra": 
                                    {"pos":111,"neg":222}
                             }

    """
    def train(self):
        pass

    def train_class():
        """
        """
        pass


    def train_positive(self):
        #freq de palavras nos textos positivos
        self.pos_freq = self.getFrequency(self.positive_corpus)
        
        # quantos documentos nessa categoria?
        self.update_doc_counts("pos",len(os.listdir(self.positive_corpus)))
        
        #atualiza no dicionario com a categoria
        self.update_counts(self.pos_freq,"pos")
   
    def train_negative(self):
        #freq de palavras nos textos negativos
        self.neg_freq = self.getFrequency(self.negative_corpus)
        
        # quantos documentos nessa categoria?
        # count_positivo count_negative
        self.update_doc_counts("neg", len(os.listdir(self.negative_corpus)))
        
        #atualiza no dicionario com a categoria
        self.update_counts(self.neg_freq,"neg")

    """ Atualiza o dicionario definido em __init__ com o numero de documentos lidos em cada categoria
    """
    def update_doc_counts(self,cat,count):
        self.doc_counts.setdefault("pos",0)
        self.doc_counts.setdefault("neg",0)
        self.doc_counts[cat]=count
        
    """ Atualiza o dicionario definido em __init__ de acordo com a frequencia da palavra e sua categoria
    """
    def update_counts(self,wordsfreq,cat):
        for word,freq in wordsfreq.iteritems():
            # entrada default
            self.counts.setdefault(word,{})
            self.counts[word].setdefault("pos",0)
            self.counts[word].setdefault("neg",0)
            # atualiza
            self.counts[word][cat]+=freq
    
    """ Calcula a frequencia de palavras para um diretorio de arquivos
    """
    def getFrequency(self,dirc):
        freq = collections.Counter()
        for filename in os.listdir(dirc):
            # frequencia de ocorrencia de cada palavra
            with open(os.path.join(dirc, filename)) as f:
                words = tokenize(f.read())
                freq.update(words)
            
        print "Diretorio: ", dirc, " / Arquivos: ", len(os.listdir(dirc)), " / Palavras unicas: ", len(freq)
        return freq

    """ Faz o calculo de probabilidades para determinar o quanto cada palavra pode ser positiva ou negativa
    """
    def calculate_probabilities(self):
        weight=1.0
        ap=0.5
        
        for word,freq in self.counts.iteritems():
            fprob_pos = float(freq["pos"]) / float(self.doc_counts["pos"])
            fprob_neg = float(freq["neg"]) / float(self.doc_counts["neg"])
            
            sum_words = freq["pos"] + freq["neg"]  
            
            # ajustando entrada para o dicionario de probabilidades
            self.probs.setdefault(word,{})       
            self.probs[word].setdefault("pos",0)
            self.probs[word].setdefault("neg",0)
            
            # formula = ((weight*ap)+(totals*basicprob))/ (weight+totals)
            self.probs[word]["pos"] = ((weight * ap) + (sum_words + fprob_pos)) / (weight + sum_words)
            self.probs[word]["neg"] = ((weight * ap) + (sum_words + fprob_neg)) / (weight + sum_words)
            
            
    """ Operacoes de classificacao
        Recebe um documento e determina sua categoria.
        TODO: otimizar 
    """
    def classifier(self, doc):
        # extrair todas as palavras do documento
        doc_words = re.findall('\w+', open(doc).read().lower())
        doc_words = collections.Counter(doc_words)

        # para cada palavra do doc. procura em self.probs e salva
        doc_prob = {}
        for word, prob in self.probs.iteritems():
            if doc_words.has_key(word):
                doc_prob.setdefault(word,prob)
        
        """ Bayes:
            Pr(Cat|Doc) = Pr(Doc|Cat) Pr(Cat)  = log Pr(Doc|Cat) + log Pr(Cat) 
                          -------------------
                                Pr(Doc)

            Pr(Doc|Cat) = mult. as prob. de cada palavra para a categoria
            
            Pr(Cat) = (n de docs na cat)
                      ------------------ 
                        total de docs

            Pr(Doc) = eliminado
        """
        
        # Pr(Doc|Cat) ...
        pmult_pos = 0
        pmult_neg = 0
        for word,prob in doc_prob.iteritems():
            # trocando multiplicacao de probabilidades por soma de logaritmos
            pmult_pos += math.log(prob["pos"])
            pmult_neg += math.log(prob["neg"])
            
        total_doc = self.doc_counts["pos"] + self.doc_counts["neg"]
        # Pr(cat = POSITIVO| Doc)
        prob_doc_pos = math.log(float(self.doc_counts["pos"]) / float(total_doc)) + pmult_pos
        # Pr(cat = NEGATIVO| Doc)
        prob_doc_neg = math.log(float(self.doc_counts["neg"]) / float(total_doc)) + pmult_neg 
   
        #resultado...
        if prob_doc_pos > prob_doc_neg: 
            return "pos"
        else:
            return "neg"
