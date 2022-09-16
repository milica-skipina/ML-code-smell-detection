# ML-code-smell-detection
This repository contains the reproducibility package for the paper "Automatic detection of Feature Envy and Data Class code smells using machine learning techniques". We used the MLCQ dataset for Data Class and Feature Envy code smell detection in our experiments:

Madeyski, L. and Lewowski, T., 2020. MLCQ: Industry-relevant code smell data set. In Proceedings of the Evaluation and Assessment in Software Engineering (pp. 342-347).

publicly available at https://zenodo.org/record/3666840#.YnOJ1ehBwuU. 

## Dataset
A dataset containing code snippets annotated for the presence of Feature Envy and Data Class code smells from the MLCQ dataset that were available for download:
* [Feature Envy](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/feature_envy.csv)
* [Data Class](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/data_class.csv)

Dataset has been divided into the training (80%) and test (20%) datasets via a stratified random sampling strategy. Each experiment has been repeated 51 times on different train-test dataset splits ([feature envy](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/Train_Test_Split.ipynb) and [data class](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/Train_Test_Split.ipynb) Jupyter notebooks) in order to get more reliable results. These train-test dataset splits can be found:
* [Feature Envy](https://github.com/milica-skipina/ML-code-smell-detection/tree/main/feature_envy/data/data_splits)
* [Data Class](https://github.com/milica-skipina/ML-code-smell-detection/tree/main/data_class/data/data_splits)

## Features extraction
We extracted the following features:
* Source code metrics – we extracted the metrics values by using the following metric extraction tools:
  * [CK Tool](https://github.com/mauricioaniche/ck/)
  * [RepositoryMiner](https://github.com/antoineBarbez/RepositoryMiner/).

   We provide two csv files with original metrics values:
    * [Feature Envy](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/embedded_datasets/metrics_dataset.csv)
    * [Data Class](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/embedded_datasets/metrics_dataset.csv)
  
* CuBERT neural source code embeddings – we used the pre-trained Java model available [here](https://github.com/google-research/google-research/tree/master/cubert).

  We extracted the 1024-dim vectors for Data Class and Feature Envy code snippets from the MLCQ dataset. First, we calculated the code embedding for every line in the code snippet separately. Afterward, we used simple mathematical operations - sum and average value of all line embeddings from the code snippet. The embeddings are available in pickle DataFrames:
  * Feature Envy:
    * [CuBERT_sum](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/embedded_datasets/cubert_embedding_sum.pkl)
    * [CuBERT_avg](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/embedded_datasets/cubert_embedding_avg.pkl)
  * Data Class:
    * [CuBERT_sum](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/embedded_datasets/cubert_embedding_sum.pkl)
    * [CuBERT_avg](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/embedded_datasets/cubert_embedding_avg.pkl)

* CodeT5 neural source code embeddings - we used base and small pre-trained models available [here](https://console.cloud.google.com/storage/browser/sfr-codet5-data-research/pretrained_models;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false).

  We extracted the 768-dim (for *base* model) and 512-dim (for *small* model) vectors for Data Class and Feature Envy code snippets from the MLCQ dataset. Besides the line by line embedding, we embedded the whole class/method at once ([feature envy](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/T5_embedding.py) and [data class](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/T5_embedding.py) Jupyter notebooks) . The embeddings are available in pickle DataFrames:
  * Feature Envy:
    * CodeT5 base model 
      * [CodeT5_base_sum](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/embedded_datasets/T5_base_line_sum.pkl)
      * [CodeT5_base_avg](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/embedded_datasets/T5_base_line_avg.pkl)
      * [CodeT5_base_method](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/embedded_datasets/T5_base.pkl)
    * Code T5 small model
      * [CodeT5_small_sum](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/embedded_datasets/T5_small_line_sum.pkl)
      * [CodeT5_small_avg](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/embedded_datasets/T5_small_line_avg.pkl)
      * [CodeT5_small_method](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/data/embedded_datasets/T5_small.pkl)
  * Data Class:
    * CodeT5 base model 
      * [CodeT5_base_sum](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/embedded_datasets/T5_base_line_sum.pkl)
      * [CodeT5_base_avg](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/embedded_datasets/T5_base_line_avg.pkl)
      * [CodeT5_base_class](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/embedded_datasets/T5_base.pkl)
    * Code T5 small model
      * [CodeT5_small_sum](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/embedded_datasets/T5_small_line_sum.pkl)
      * [CodeT5_small_avg](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/embedded_datasets/T5_small_line_avg.pkl)
      * [CodeT5_small_class](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/data/embedded_datasets/T5_small.pkl)

## Results
Jupyter notebooks evaluating the performance of all approaches:
* [Feature Envy](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/results/Results.ipynb)
* [Data Class](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/results/Results.ipynb)

## Feature importance analysis
Jupyter notebooks presenting the most important features of models trained over 51 trials using source code metrics:
* [Feature Envy](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/feature_envy/results/ML_metrics_Feature_Importance.ipynb)
* [Data Class](https://github.com/milica-skipina/ML-code-smell-detection/blob/main/data_class/results/ML_metrics_Feature_Importance.ipynb)
