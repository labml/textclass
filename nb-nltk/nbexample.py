"""
Download do corpus:

http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

Arquivo com duas pastas com reviews positivos e negativos.
Funciona com a mesma ideia de um classificador de spam.
"""

import os.path
import NaiveBayes as nb

BASE_DIR = "..."


def read_test_item(item_file):
    contents = ""
    with open(item_file) as f:
        contents = f.read()
    return contents


def create_classifier():
    dir_pos = os.path.join(BASE_DIR, "pos")
    dir_neg = os.path.join(BASE_DIR, "neg")

    nbc = nb.NaiveBayes(positive_corpus=dir_pos,
                        negative_corpus=dir_neg)

    # treina as duas categorias
    nbc.train()

    return nbc


def test_classifier(nbc):
    dir_pos_teste = os.path.join(BASE_DIR, "postest")
    dir_neg_teste = os.path.join(BASE_DIR, "negtest")

    tp, fp, tn, fn = 0, 0, 0, 0

    print "Testando positivos..."
    for filename in os.listdir(dir_pos_teste):
        # envia o arquivo completo
        item_class = nbc.classifier(os.path.join(dir_pos_teste, filename))
        if item_class == 'pos':
            tp += 1
        else:
            fn += 1

    print "Testando negativos..."
    for filename in os.listdir(dir_neg_teste):
        # envia o arquivo completo
        item_class = nbc.classifier(os.path.join(dir_neg_teste, filename))
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
    nbc = create_classifier()
    test_classifier(nbc)
