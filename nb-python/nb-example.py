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

    tp, fp, tn, fn = 0, 0, 0, 0

    print "Testando positivos..."
    for filename in os.listdir(dir_pos_teste):
        item_contents = read_test_item(os.path.join(dir_pos_teste, filename))
        item_class = nb.classify_item(item_contents)
        if item_class == 'pos':
            tp += 1
        else:
            fn += 1

    print "Testando negativos..."
    for filename in os.listdir(dir_neg_teste):
        item_contents = read_test_item(os.path.join(dir_neg_teste, filename))
        item_class = nb.classify_item(item_contents)
        if item_class == 'neg':
            tn += 1
        else:
            fp += 1

    print 'Acertos: ', (tp + tn), ' / 600'
    print 'Accuracy: ', float(tp + tn) / 600.0
    print 'TP: ', tp, ' - FP: ', fp, ' - TN: ', tn, ' - FN: ', fn
    precision = float(tp) / float(tp + fp)
    print 'Precision: ', precision
    recall = float(tp) / float(tp + fn)
    print 'Recall: ', recall
    print 'F1 score: ', 2 * (precision * recall) / (precision + recall)

if __name__ == '__main__':
    nb = create_classifier()
    test_classifier(nb)
