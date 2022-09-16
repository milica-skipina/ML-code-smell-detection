# ML-code-smell-detection
This repository contains the reproducibility package for the paper "Automatic detection of Feature Envy and Data Class code smells using machine learning techniques". We used the MLCQ dataset for Data Class and Feature Envy code smell detection in our experiments:

Madeyski, L. and Lewowski, T., 2020. MLCQ: Industry-relevant code smell data set. In Proceedings of the Evaluation and Assessment in Software Engineering (pp. 342-347).

publicly available at https://zenodo.org/record/3666840#.YnOJ1ehBwuU. 

## Dataset
A dataset containing code snippets annotated for the presence of Feature Envy and Data Class code smells from the MLCQ dataset that were available for download:
* [Feature Envy](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/feature_envy.csv)
* [Data Class](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/data_class.csv)

Dataset has been divided into the training (80%) and test (20%) datasets via a stratified random sampling strategy. Each experiment has been repeated 51 times on different train-test dataset splits ([feature envy](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/Train_Test_Split.ipynb) and [data class](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/Train_Test_Split.ipynb) notebooks) in order to get more reliable results. These train-test dataset splits can be found:
* [Feature Envy](https://github.com/milica-skipina/ML-code-smell-detection/tree/main/feature_envy/data/data_splits)
* [Data Class](https://github.com/milica-skipina/ML-code-smell-detection/tree/main/data_class/data/data_splits)

## Features extraction
We extracted the following features:
* Source code metrics – we extracted the metrics values by using the following metric extraction tools:
  * [CK Tool](https://github.com/mauricioaniche/ck/)
  * [RepositoryMiner](https://github.com/antoineBarbez/RepositoryMiner/).
* CuBERT neural source code embeddings – we used the pre-trained Java model available [here](https://github.com/google-research/google-research/tree/master/cubert).
* CodeT5 neural source code embeddings - we used base and small pre-trained models available [here](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/pretrained_models;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
