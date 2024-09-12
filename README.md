# LLM Topic Extraction for Conversation Analytics

## Overview
This capsule contains the code to execute the experiments to compare the performance impact of using non-LLM-based vs. LLM-based text representations in supervised classification and unsupervised document clustering topic extraction approaches for analyzing conversational text data.
Each configuration contains three components: text representation, dimensionality reduction, and topic extraction.
Our full experiment includes a total of 160 non-LLM configurations and 320 LLM configurations per dataset with a choice of the following components:
> - four non-LLM-based text representations: GloVe, Komininos, Levy and TF-IDF
> - eight LLM-based text representations: DistilRoBERTa, MiniLM, MPNet, RoBERTa, GPT Ada, GPT Curie, GPT Davinci, USE
> - five dimensionality reduction approaches: No DR, ICA, PCA, UMAP and SVD
> - four supervised classification approaches: LGBM, LR, RF, XGBoost
> - four unsupervised classification approaches: BIRCH, HDBSCAN, AHC, k-means
> - three datasets: banking77, clinc150 and hwu64

## Contents
environment: contains the Dockerfile to build the environment with the required package
code: 
> - _run_one.py_ executes the experiment for one specific configuration
> - _run_all.py_ executes multiple experiments in one shot and calls run_one.py for each configuration
> - _run_ is the master script for the capsule and calls _run_all.py_ to execute the experiment
data: Due to space limitation, we have included the raw text data file from banking77 and the embeddings from MPNet for this dataset.
This will allow the reproducibility team to execute the experiments of any configurations for banking77 using TF-IDF or MPNet with any dimensionality reduction approaches and supervised or unsupervised topic extraction approaches.

