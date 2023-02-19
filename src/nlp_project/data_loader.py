import os
import sys
import os.path as op
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split


def get_imdb_dataset():
    """
    This function gets the imdb dataset from the path:./aclImdb/train/neg for negative reviews, and./aclImdb/train/pos for positive reviews
    """
    # We get the files from the path: ./aclImdb/train/neg for negative reviews, and ./aclImdb/train/pos for positive reviews
    imdb_train_filenames_neg = sorted(
        glob(op.join(".", "aclImdb", "train", "neg", "*.txt"))
    )
    imdb_train_filenames_pos = sorted(
        glob(op.join(".", "aclImdb", "train", "pos", "*.txt"))
    )

    imdb_test_filenames_neg = sorted(
        glob(op.join(".", "aclImdb", "test", "neg", "*.txt"))
    )
    imdb_test_filenames_pos = sorted(
        glob(op.join(".", "aclImdb", "test", "pos", "*.txt"))
    )

    # Each files contains a review that consists in one line of text: we put this string in two lists, that we concatenate
    imdb_train_texts_neg = [
        open(f, encoding="utf8").read() for f in imdb_train_filenames_neg
    ]
    imdb_train_texts_pos = [
        open(f, encoding="utf8").read() for f in imdb_train_filenames_pos
    ]
    imdb_train_texts = imdb_train_texts_neg + imdb_train_texts_pos

    imdb_test_texts_neg = [
        open(f, encoding="utf8").read() for f in imdb_test_filenames_neg
    ]
    imdb_test_texts_pos = [
        open(f, encoding="utf8").read() for f in imdb_test_filenames_pos
    ]
    imdb_test_texts = imdb_test_texts_neg + imdb_test_texts_pos

    # The first half of the elements of the list are string of negative reviews, and the second half positive ones
    # We create the labels, as an array of [1,len(texts)], filled with 1, and change the first half to 0
    imdb_train_labels = np.ones(len(imdb_train_texts), dtype=int)
    imdb_train_labels[: len(imdb_train_texts_neg)] = 0.0

    imdb_test_labels = np.ones(len(imdb_test_texts), dtype=int)
    imdb_test_labels[: len(imdb_test_texts_neg)] = 0.0

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        imdb_train_texts, imdb_train_labels, test_size=0.2
    )

    return train_texts, test_texts, train_labels, test_labels


def get_movie_review_dataset():
    """
    Get the movie review dataset
    """
    movie_review_filenames_neg = sorted(
        glob(op.join(".", "movie_reviews", "movie_reviews", "neg", "*.txt"))
    )
    movie_review_filenames_pos = sorted(
        glob(op.join(".", "movie_reviews", "movie_reviews", "pos", "*.txt"))
    )

    # Each files contains a review that consists in one line of text: we put this string in two lists, that we concatenate
    movie_review_texts_neg = [
        open(f, encoding="ascii").read() for f in movie_review_filenames_neg
    ]
    movie_review_texts_pos = [
        open(f, encoding="utf8").read() for f in movie_review_filenames_pos
    ]
    movie_review_texts = movie_review_texts_neg + movie_review_texts_pos
    movie_review_texts = list(map(lambda x: x.replace("\n", ""), movie_review_texts))

    # The first half of the elements of the list are string of negative reviews, and the second half positive ones
    # We create the labels, as an array of [1,len(texts)], filled with 1, and change the first half to 0
    movie_review_labels = np.ones(len(movie_review_texts), dtype=int)
    movie_review_labels[: len(movie_review_texts_neg)] = 0.0

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        movie_review_texts, movie_review_labels, test_size=0.2
    )

    return train_texts, test_texts, train_labels, test_labels


def get_reuters_dataset():
    reuters_filename_test = sorted(glob(op.join(".", "reuters", "reuters", "test", "*")))
    reuters_filename_train = sorted(glob(op.join(".", "reuters", "reuters", "training", "*")))
    cats = open(op.join(".", "reuters", "reuters", "cats.txt"), encoding="ascii").read()
    M = np.array([cat.split("/") for cat in cats.split('\n')[:-1]])
    cats_dict = {int(d[0]): d[1:] for d in [m.split(' ') for m in M[:, 1]]}

    nb_e = 0
    reuters_texts_test = []
    reuters_labels_test = []
    for f in reuters_filename_test:
        try :
          reuters_texts_test.append(open(f, encoding="ascii").read().replace("\n", ""))
          reuters_labels_test.append(cats_dict[int(f.split("/")[-1])])
        except :
          nb_e += 1
    print(f"{str(nb_e)} file(s) in test aren't downloaded because of encoding type")

    nb_e = 0
    reuters_texts_train = []
    reuters_labels_train = []
    for f in reuters_filename_train:
        try :
          reuters_texts_train.append(open(f, encoding="ascii").read().replace("\n", ""))
          reuters_labels_train.append(cats_dict[int(f.split("/")[-1])])
        except :
          nb_e += 1
    print(f"{str(nb_e)} file(s) in train aren't downloaded because of encoding type")

    return reuters_texts_train, reuters_texts_test, reuters_labels_train, reuters_labels_test