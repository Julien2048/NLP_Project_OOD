# Detecting Textual Out-Of-Distribution Examples with Pretrained Transformers

This project is part of the final year NLP course at ENSAE. We have written a paper on this subject and here is the abstract.

Deep learning models allow today to train large datasets in order to have very good performance in all problems that call for Natural Language Processing (NLP). Nevertheless, these models are very vulnerable to the change in distribution between the training dataset and the application data. This is why it has become essential to develop methods to detect data that are not of the same distribution. In this paper, we give a quick background of the out-of-distribution problem. Then, we present different detectors, starting with their mathematical basis, and we change the DistilBERT model in order to have different aggregations. Finally, we propose to apply the detectors to several datasets and aggregation in order to compare them.

## Requirements

To create the environment to work on, on a terminal, use:

```
$ conda env create -f environment.yml
```

or to install packages, use:
```
$ pip install -r requirements.txt
```

## Usage example
 
You can create a folder on you drive to keep intermediate data to easily use it.

The ```models_scripts``` folder contains the modification files of the ```transformers``` package to access the outputs of the intermediate layers of the DistilBERT model, in order to use the created classes, you will have to use them to replace the initial corresponding files of your transformers package.

You can find several examples of use in the ```Notebook``` folder.
For instance for IMDB as in-distribution, you will have to run the ```get_data_train_model_imdb.ipynb```to download the data and trained the model. And then, you can use the ```get_metric_ood_imdb.ipynb```notebook to compute the OOD detections.


