# Detecting Out-Of-Distribution Examples with Pretrained Transformers

This project is part of the final year NLP course at ENSAE. We have written a paper on this subject and here is the abstract.

Deep learning models allow today to train large datasets in order to have very good performance in all problems that call for Natural Language Processing (NLP). Nevertheless, these models are very vulnerable to the change in distribution between the training dataset and the application data. This is why it has become essential to develop methods to detect data that are not of the same distribution. In this paper, we give a quick background of the out-of-distribution problem. Then, we present different detectors, starting with their mathematical basis, and we change the DistilBERT model in order to have different aggregations. Finally, we propose to apply the detectors to several datasets and aggregation in order to compare them.

## Requirement

To create the environment to work on, on a terminal, use :

```
$ conda env create -f environment.yml
```

## Usage example

You can find several examples of use in the ```Notebook``` folder.
You can also find the command lines to download the main datasets we used in ```Notebook```.

The ```models_scripts``` folder contains the modification files of the ```transformers``` package to access the outputs of the intermediate layers of the DistilBERT model.
