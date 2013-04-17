"""
Download do corpus:

http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

Arquivo com duas pastas com reviews positivos e negativos.
Funciona com a mesma ideia de um classificador de spam.
"""

import os.path
import NaiveBayesClassifier as NB


def read_test_item(item_file):
    contents = ""
    with open(item_file) as f:
        contents = f.read()
    return contents


def create_classifier():
    dir_pos = "pos"
    dir_neg = "neg"

    nb = NB.NaiveBayesClassifier(positive_corpus=dir_pos, negative_corpus=dir_neg)

    nb.train_positive()
    nb.train_negative()

    # cria um dicionario com as probabilidades de cada palavra
    nb.calculate_probabilities()

    return nb


def test_classifier(nb):
    dir_pos_teste = "postest"
    dir_neg_teste = "negtest"

    # classificador = nb.classifier(dir_pos_teste, dir_neg_teste)

    print "Testando positivos:"
    for filename in os.listdir(dir_pos_teste):
        item_contents = read_test_item(os.path.join(dir_pos_teste, filename))
        item_class = nb.classify_item(item_contents)
        print 'Classe: ', item_class

    print "Testando negativos:"
    for filename in os.listdir(dir_neg_teste):
        item_contents = read_test_item(os.path.join(dir_neg_teste, filename))
        item_class = nb.classify_item(item_contents)
        print 'Classe: ', item_class


if __name__ == '__main__':
    nb = create_classifier()
    test_classifier(nb)
