import os.path as op
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
import random
from sklearn.datasets import fetch_20newsgroups


class IMDBDataset:
    def __init__(self):
        pass

    def download_imdb_dataset(self, google_colab=True):
        if google_colab:
            pass

    def get_dataset(self):
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

        (
            self.train_texts,
            self.test_texts,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(
            imdb_train_texts, imdb_train_labels, test_size=0.2, random_state=42
        )

        return self.train_texts, self.test_texts, self.train_labels, self.test_labels

    def save_labels(self):
        np.save("imdb_train_labels.npy", self.train_labels)
        np.save("imdb_test_labels.npy", self.test_labels)

    def save_texts(self):
        np.save("imdb_train_texts.npy", self.train_texts)
        np.save("imdb_test_texts.npy", self.test_texts)

    def load_labels(self, path=""):
        imdb_train_labels = np.load(path + "imdb_train_labels.npy")
        imdb_test_labels = np.load(path + "imdb_test_labels.npy")
        return imdb_train_labels, imdb_test_labels

    def load_texts(self, path=""):
        imdb_train_texts = np.load(path + "imdb_train_texts.npy")
        imdb_test_texts = np.load(path + "imdb_test_texts.npy")
        return imdb_train_texts, imdb_test_texts


class MovieReviewDataset:
    def __init__(self):
        pass

    def download_dataset(self, kaggle_json=True):
        if kaggle_json:
            pass
        else:
            print("Upload kaggle.json in content")

    def get_dataset(self):
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
        movie_review_texts = list(
            map(lambda x: x.replace("\n", ""), movie_review_texts)
        )

        # The first half of the elements of the list are string of negative reviews, and the second half positive ones
        # We create the labels, as an array of [1,len(texts)], filled with 1, and change the first half to 0
        movie_review_labels = np.ones(len(movie_review_texts), dtype=int)
        movie_review_labels[: len(movie_review_texts_neg)] = 0.0

        (
            self.train_texts,
            self.test_texts,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(movie_review_texts, movie_review_labels, test_size=0.2)

        return self.train_texts, self.test_texts, self.train_labels, self.test_labels

    def save_labels(self):
        np.save("movie_review_train_labels.npy", self.train_labels)
        np.save("movie_review_test_labels.npy", self.test_labels)

    def save_texts(self):
        np.save("movie_review_train_texts.npy", self.train_texts)
        np.save("movie_review_test_texts.npy", self.test_texts)

    def load_labels(self, path=""):
        movie_review_train_labels = np.load(path + "movie_review_train_labels.npy")
        movie_review_test_labels = np.load(path + "movie_review_test_labels.npy")
        return movie_review_train_labels, movie_review_test_labels

    def load_texts(self, path=""):
        movie_review_train_texts = np.load(path + "movie_review_train_texts.npy")
        movie_review_test_texts = np.load(path + "movie_review_test_texts.npy")
        return movie_review_train_texts, movie_review_test_texts


class MNLIDataset:
    def __init__(self):
        pass

    def get_dataset(self):
        mnli_mismatched = load_dataset("glue", "mnli_mismatched", split="test")
        mnli_matched = load_dataset("glue", "mnli_matched", split="test")

        out_mnli_mismatched = mnli_mismatched["premise"]
        out_mnli_matched = mnli_matched["premise"]

        self.test_texts = out_mnli_mismatched + out_mnli_matched
        random.shuffle(self.test_texts)

        return self.test_texts

    def save_texts(self):
        np.save("mnli_test_texts.npy", self.test_texts)

    def load_texts(self, path=""):
        return np.load(path + "mnli_test_texts.npy")


class ReutersDataset:
    def __init__(self):
        target_names = [
            "alt.atheism",
            "comp.graphics",
            "comp.os.ms-windows.misc",
            "comp.sys.ibm.pc.hardware",
            "comp.sys.mac.hardware",
            "comp.windows.x",
            "misc.forsale",
            "rec.autos",
            "rec.motorcycles",
            "rec.sport.baseball",
            "rec.sport.hockey",
            "sci.crypt",
            "sci.electronics",
            "sci.med",
            "sci.space",
            "soc.religion.christian",
            "talk.politics.guns",
            "talk.politics.mideast",
            "talk.politics.misc",
            "talk.religion.misc",
        ]

        self.in_target_names = target_names[:15]
        self.out_target_names = target_names[15:]

    def get_dataset(self):
        train_in = fetch_20newsgroups(subset="train", categories=self.in_target_names)
        test_in = fetch_20newsgroups(subset="test", categories=self.in_target_names)
        test_out = fetch_20newsgroups(subset="test", categories=self.out_target_names)

        self.in_newsgroups_train_texts = [
            text.replace("\n", " ").replace("\t", " ").replace(">", "")
            for text in train_in.data
        ]
        self.in_newsgroups_train_labels = train_in.target
        self.in_newsgroups_test_texts = [
            text.replace("\n", " ").replace("\t", " ").replace(">", "")
            for text in test_in.data
        ]
        self.in_newsgroups_test_labels = test_in.target

        self.out_newsgroups_test_texts = [
            text.replace("\n", " ").replace("\t", " ").replace(">", "")
            for text in test_out.data
        ]
        self.out_newsgroups_test_labels = test_out.target

        return (
            self.in_newsgroups_train_texts,
            self.in_newsgroups_test_texts,
            self.out_newsgroups_test_texts,
            self.in_newsgroups_train_labels,
            self.in_newsgroups_test_labels,
            self.out_newsgroups_test_labels,
        )

    def save_texts(self):
        np.save("reuters_in_train_texts.npy", self.in_newsgroups_train_texts)
        np.save("reuters_in_test_texts.npy", self.in_newsgroups_test_texts)
        np.save("reuters_out_test_texts.npy", self.out_newsgroups_test_texts)

    def save_labels(self):
        np.save("reuters_in_train_labels.npy", self.in_newsgroups_train_labels)
        np.save("reuters_in_test_labels.npy", self.in_newsgroups_test_labels)
        np.save("reuters_out_test_labels.npy", self.out_newsgroups_test_labels)

    def load_texts(self, path=""):
        in_newsgroups_train_texts = np.load(path + "reuters_in_train_texts.npy")
        in_newsgroups_test_texts = np.load(path + "reuters_in_test_texts.npy")
        out_newsgroups_test_texts = np.load(path + "reuters_out_test_texts.npy")
        return (
            in_newsgroups_train_texts,
            in_newsgroups_test_texts,
            out_newsgroups_test_texts,
        )

    def load_labels(self, path=""):
        in_newsgroups_train_labels = np.load(path + "reuters_in_train_labels.npy")
        in_newsgroups_test_labels = np.load(path + "reuters_in_test_labels.npy")
        out_newsgroups_test_labels = np.load(path + "reuters_out_test_labels.npy")
        return (
            in_newsgroups_train_labels,
            in_newsgroups_test_labels,
            out_newsgroups_test_labels,
        )


class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_reuters_dataset():
    reuters_filename_test = sorted(
        glob(op.join(".", "reuters", "reuters", "test", "*"))
    )
    reuters_filename_train = sorted(
        glob(op.join(".", "reuters", "reuters", "training", "*"))
    )
    cats = open(op.join(".", "reuters", "reuters", "cats.txt"), encoding="ascii").read()
    M = np.array([cat.split("/") for cat in cats.split("\n")[:-1]])
    cats_dict = {int(d[0]): d[1:] for d in [m.split(" ") for m in M[:, 1]]}

    nb_e = 0
    reuters_texts_test = []
    reuters_labels_test = []
    for f in reuters_filename_test:
        try:
            reuters_texts_test.append(
                open(f, encoding="ascii").read().replace("\n", "")
            )
            reuters_labels_test.append(cats_dict[int(f.split("/")[-1])])
        except:
            nb_e += 1
    print(f"{str(nb_e)} file(s) in test aren't downloaded because of encoding type")

    nb_e = 0
    reuters_texts_train = []
    reuters_labels_train = []
    for f in reuters_filename_train:
        try:
            reuters_texts_train.append(
                open(f, encoding="ascii").read().replace("\n", "")
            )
            reuters_labels_train.append(cats_dict[int(f.split("/")[-1])])
        except:
            nb_e += 1
    print(f"{str(nb_e)} file(s) in train aren't downloaded because of encoding type")

    return (
        reuters_texts_train,
        reuters_texts_test,
        reuters_labels_train,
        reuters_labels_test,
    )
