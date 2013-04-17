import random
import math
import urllib
import glob
import subprocess
import os.path
import shutil

# dictionary listing datasets
corpora_url = {
    'reviews': 'http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz'
}


def expand_glob(glob_ptn):
    exps = glob.glob(glob_ptn)
    if len(exps) < 1:
        return None
    else:
        return exps[0]


def move_numbered_files(numbers, sourcedir, destdir):
    for i in numbers:
        name_glob = 'cv{0:03d}_*.txt'.format(i)
        print 'Expanding ' + name_glob
        fname = expand_glob(os.path.join(sourcedir, name_glob))
        shutil.move(fname, os.path.join(destdir, os.path.basename(fname)))
    return


# assumes that the program is in the base directory of review corpus
def prepare_review_corpus(test_proportion, seed):
    nreviews = 1000
    review_ns = set(range(0, nreviews))
    random.seed(seed)
    train_pos, test_pos = partition_traintest(review_ns, test_proportion)
    train_neg, test_neg = partition_traintest(review_ns, test_proportion)
    pos, neg = "pos/", "neg/"
    postest, negtest = "postest", "negtest"
    os.mkdir(postest)
    move_numbered_files(test_pos, pos, postest)
    os.mkdir(negtest)
    move_numbered_files(test_neg, neg, negtest)
    return


# partitions the cases into training and testing (selected at random)
# proportion is the proportion of cases in test
# assume balanced classes
# cases must be a set
def partition_traintest(cases, proportion):
    ntest = math.trunc(math.floor(len(cases) * proportion))
    testcases = set(random.sample(cases, ntest))
    traincases = cases.difference(testcases)
    return traincases, testcases


def download_file(fileurl, localfile):
    return urllib.urlretrieve(fileurl, localfile)


def prepare_corpus(corpus_url, test_proportion, seed):
    print 'Downloading ' + corpus_url + ' ...'
    local_fname = os.path.basename(corpus_url)
    fname, hdr = urllib.urlretrieve(corpus_url, local_fname)
    print 'Extracting ' + local_fname
    res = subprocess.call(['tar', 'zxvf', local_fname])
    if res == 0:
        print 'Alles ok!'
    else:
        print 'Schwartzkopf ist Kaputt!'
    return

if __name__ == "__main__":
    seed = 11313717   # reuse seed to reproduce results
    prepare_review_corpus(0.3, seed)
