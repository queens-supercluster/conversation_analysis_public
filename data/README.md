# Data

This directory holds the raw datasets as well as scripts to clean and process the raw datasets.

## The Datasets

Each dataset is contained in a subdirector of this directory, in a CSV file.

## Dataset EDA

The script `do_eda.py` can be run to read in all the datasets and output some simple EDA/statistics to a file named `datasets.md`.

## Cleaning

Each subdirectory has a script called `make_clean.py` that will take the raw dataset, perform various cleaning steps, and output a new file named `clean.csv`. (In the case of `banking77`, no cleaning was necessary.)

## Embeddings

Because getting the embeddings for each dataset is a slow process, and we don't want to do it for each experiment trial, we have a script that can compute the embeddings once and save them as a CSV file.

The script `get_embeddings.py` will take the clean version of a dataset and calculate many different embeddings (e.g., USE, GPT, Pooling, SBert) and save the results in a new CSV file that can then be used by the experiment scripts.