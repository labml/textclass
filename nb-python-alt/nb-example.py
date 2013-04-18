"""
Download do corpus:

https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/movie_reviews.zip

Arquivo com duas pastas com reviews positivos e negativos. Funciona com a mesma ideia de um classificador de spam.
"""

import NaiveBayesClassifier

# Para treinar ... (exemplo usado com 700 arquivos de texto em cada pasta)

dir_pos = "diretorio/pos/"
dir_neg = "diretorio/neg/"

nb = NaiveBayesClassifier(positive_corpus=dir_pos,negative_corpus=dir_neg)

nb.train_positive()
nb.train_negative()

#cria um dicionario com as probabilidades positivas e negativas de cada palavra
#e será utilizado para os cálculos futuros
nb.calculate_probabilities()


# Para classificação/teste (exemplo usado de 300 arquivos de texto em cada pasta)
dir_pos_teste = "diretorio/teste/pos/"
dir_neg_teste = "diretorio/teste/neg/"

# Testando. Para cada documento do diretÛrio, verifica a classificacao 
for filename in os.listdir(dir_pos_teste):
    print nb.naive_bayes(dir_pos_teste + filename)


