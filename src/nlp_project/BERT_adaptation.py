import numpy as np
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassificationPreLogits
from transformers import DistilBertForSequenceClassificationHiddenLayer
from transformers import Trainer, TrainingArguments
import torch


class TokenizeData:
    def __init__(self, model="DistilBert"):
        if model == "DistilBert":
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                "distilbert-base-uncased"
            )
        if not model:
            pass

    def __call__(self, texts, name_data, name_splt, nb_texts=None):
        self.name_data = name_data
        self.name_splt = name_splt

        encodings = self.tokenize_texts(texts, nb_texts)

        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

        return self.input_ids, self.attention_mask

    def tokenize_texts(self, texts, nb_texts=None):
        if not nb_texts:
            nb_texts = len(texts)
        else:
            nb_texts = min(len(texts), nb_texts)

        # Tokenize the input sentences
        encodings = self.tokenizer(
            texts[:nb_texts],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encodings

    def save_tokens(self):
        np.save(
            self.name_data + "_input_ids_" + self.name_splt + ".npy", self.input_ids
        )
        np.save(
            self.name_data + "_attention_mask_" + self.name_splt + ".npy",
            self.attention_mask,
        )

    def load_tokens(self, name_data, name_splt, path=""):
        input_ids = np.load(path + name_data + "_input_ids_" + name_splt + ".npy")
        attention_mask = np.load(
            path + name_data + "_attention_mask_" + name_splt + ".npy"
        )
        return input_ids, attention_mask


class DistilBertClassifier:
    def __init__(
        self,
        device,
        prelogits=True,
        path_pretrained_model="distilbert-base-uncased",
        num_labels=2,
        batch_size=32,
        weight_decay=0.01,
        warmup_ratio=0.06,
        learning_rate=1e-5,
        num_epochs=3,
        log_steps=100,
    ):
        self.device = device
        self.BATCH_SIZE = batch_size
        self.WEIGHT_DECAY = weight_decay
        self.WARMUP_RATIO = warmup_ratio
        self.LEARNING_RATE = learning_rate
        self.NUM_EPOCHS = num_epochs
        self.LOG_STEPS = log_steps
        self.num_labels = num_labels
        self.prelogits = prelogits

        if self.prelogits:
            self.model = DistilBertForSequenceClassificationPreLogits.from_pretrained(
                path_pretrained_model,
                output_hidden_states=True,
                num_labels=self.num_labels,
            ).to(self.device)
        else:
            self.model = DistilBertForSequenceClassificationHiddenLayer.from_pretrained(
                path_pretrained_model,
                output_hidden_states=True,
                num_labels=self.num_labels,
            ).to(self.device)

    def train_model(self, train_dataset, test_dataset):
        # Define the training hyperparameters
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=self.BATCH_SIZE,
            per_device_eval_batch_size=self.BATCH_SIZE,
            num_train_epochs=self.NUM_EPOCHS,
            weight_decay=self.WEIGHT_DECAY,
            warmup_ratio=self.WARMUP_RATIO,
            logging_steps=self.LOG_STEPS,
            learning_rate=self.LEARNING_RATE,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        trainer.train()

    def save_pretrained_model(self, name_model, path="model_trained/"):
        self.model.save_pretrained(path + name_model)

    def get_prelogit_logit(
        self,
        input_ids,
        attention_mask,
        name_data=None,
        name_splt=None,
        size_array=100,
        save=False,
    ):
        if not self.prelogits:
            raise ValueError("The model is not a prelogits model")

        nb_obs = len(input_ids)

        # Split the test set into batches
        prelogits_array = np.empty(shape=(nb_obs, 768))
        logits_array = np.empty(shape=(nb_obs, self.num_labels))

        with torch.no_grad():
            for i in range(0, nb_obs, size_array):
                input_ids_batch = input_ids[i : i + size_array]
                attention_mask_batch = attention_mask[i : i + size_array]

                outputs = self.model(
                    input_ids=input_ids_batch, attention_mask=attention_mask_batch
                )
                logits = outputs[0]
                prelogits = outputs[1]

                prelogits_array[i : i + size_array] = prelogits.cpu()
                logits_array[i : i + size_array] = logits.cpu()

        if save:
            np.save(name_data + "_prelogits_" + name_splt + ".npy", prelogits_array)
            np.save(name_data + "_logits_" + name_splt + ".npy", logits_array)

        return prelogits_array, logits_array

    def load_prelogit_logit(self, name_data, name_splt, path: str = ""):
        prelogits = np.load(path + name_data + "_prelogits_" + name_splt + ".npy")
        logits = np.load(path + name_data + "_logits_" + name_splt + ".npy")
        return prelogits, logits

    def get_hidden_layer(
        self,
        input_ids,
        attention_mask,
        name_data=None,
        name_splt=None,
        size_array=100,
        save=False,
    ):
        if self.prelogits:
            raise ValueError("The model is not a hidden layer model")

        nb_obs = len(input_ids)

        # Split the test set into batches
        hidden_layer_array = np.empty(shape=(nb_obs, 768))

        with torch.no_grad():
            for i in range(0, nb_obs, size_array):
                input_ids_batch = input_ids[i : i + size_array]
                attention_mask_batch = attention_mask[i : i + size_array]

                outputs = self.model(
                    input_ids=input_ids_batch, attention_mask=attention_mask_batch
                )
                hidden_layer = outputs["hidden_states"]

                hidden_layer_array[i : i + size_array] = hidden_layer.cpu()

        if save:
            np.save(
                name_data + "_hidden_layer_" + name_splt + ".npy", hidden_layer_array
            )

        return hidden_layer_array

    def load_hidden_layer(self, name_data, name_splt, path: str = ""):
        hidden_layer = np.load(path + name_data + "_hidden_layer_" + name_splt + ".npy")
        return hidden_layer
