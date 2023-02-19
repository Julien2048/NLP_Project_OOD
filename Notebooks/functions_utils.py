import glob
import numpy as np
import os.path as op
from sklearn.model_selection import train_test_split
import torch


def get_imdb_dataset():
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


# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def tokenize_texts(tokenizer, texts, length_texts=None):
    if not length_texts:
        length_texts = len(texts)
    else:
        length_texts = min(len(texts), length_texts)
    # Tokenize the input sentences
    encoded = tokenizer.batch_encode_plus(
        texts[:length_texts],
        add_special_tokens=True,
        # max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Get the input IDs, attention masks, and token type IDs
    input_ids = encoded["input_ids"].to(device)
    attention_masks = encoded["attention_mask"].to(device)
    token_type_ids = encoded["token_type_ids"].to(device)

    return input_ids, attention_masks, token_type_ids


def get_prelogits(model, input_ids, attention_masks, token_type_ids):
    # Get the prelogits of the train dataset
    with torch.no_grad():
        outputs = model(input_ids, attention_masks, token_type_ids)
        prelogits = outputs[1].cpu().numpy()

    return prelogits
