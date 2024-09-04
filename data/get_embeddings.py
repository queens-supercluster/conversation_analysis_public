import argparse
import numpy as np
import pandas as pd
import os
import joblib
import sys
import optuna
from pathlib import Path

from sentence_transformers import SentenceTransformer, LoggingHandler
import tensorflow_hub as hub

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--project-dir', type = str, default="/global/project/hpcg1614_shared/ca/")
args = parser.parse_args()

all = []

for dataset in ['banking77', 'clinc150', 'hwu64', 'amazon_all', 'test3' ]:

    if dataset == "banking77":
        train_fn = "{}/data/banking77/train.csv".format(args.project_dir)
        text_col = "text"
        target_col = "category"
    elif dataset == "amazon_all":
        train_fn = "{}/data/amazon-all/clean.csv".format(args.project_dir)
        text_col = "utterance"
        target_col = "intent"
    elif dataset == "clinc150":
        train_fn = "{}/data/clinc150/clean.csv".format(args.project_dir)
        text_col = "text"
        target_col = "label"
    elif dataset == "hwu64":
        train_fn = "{}/data/hwu64/clean.csv".format(args.project_dir)
        text_col = "text"
        target_col = "label"
    elif dataset == "test3":
        train_fn = "{}/data/test3/clean.csv".format(args.project_dir)
        text_col = "text"
        target_col = "category"

    train_df = pd.read_csv(train_fn)
        
    models =  [
        #"allenai-specter",
        "all-mpnet-base-v2", 
        "multi-qa-mpnet-base-dot-v1",
        #"facebook-dpr-ctx_encoder-multiset-base",
        #"facebook-dpr-question_encoder-multiset-base",
        "all-distilroberta-v1", 
        "msmarco-distilbert-cos-v5",
        "all-MiniLM-L12-v2", 
        "multi-qa-distilbert-cos-v1",
        'all-MiniLM-L6-v2', 
        "all-roberta-large-v1",
        "distilbert-base-nli-stsb-quora-ranking",
        #"paraphrase-albert-small-v2",
        #"paraphrase-MiniLM-L3-v2",
        "average_word_embeddings_glove.6B.300d",
        #"average_word_embeddings_komninos",
        #"average_word_embeddings_levy_dependency",
        #"msmarco-bert-co-condensor",
        #"stsb-roberta-base-v2",
        #"nq-distilbert-base-v1",
        #'gpt2-medium',
        #"albert-base-v2",
        #"t5-base",
        "universal-sentence-encoder",
        #"text-similarity-ada-001",
        #"text-similarity-babbage-001",
        #"text-similarity-curie-001",
        "text-similarity-davinci-001",
    ]
    #models =  [
        #"text-similarity-ada-001",
    #]
        
    for model in models:
        print("\nDataset={}, model={}".format(dataset, model))
        dirname = os.path.dirname(train_fn)
        new_fn = os.path.join(dirname, "clean_embed_{}.csv".format(model))
        if os.path.isfile(new_fn):
            print("File already exists! Skipping.")
            continue

        X = train_df[text_col].to_numpy()
       
        try:
            if model == "universal-sentence-encoder":
                embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                train_embeddings = np.array(embedder(X))
            elif "text-similarity" in model:
                import openai
                openai.api_key_path="key.txt"
                
                X = list(X)
                
                train_embeddings = []
                cur = 0
                for x in X:
                    cur = cur + 1
                    print("{} / {}".format(cur, len(X)))
                    #print(x)
                    #print(len(x))
                    #if cur > 292:
                        #break
                    e = openai.Embedding.create(input=x, engine=model)
                    train_embeddings.append(e['data'][0]['embedding'])
                
            else:
                embedder = SentenceTransformer(model_name_or_path=model)
                train_embeddings = embedder.encode(X, convert_to_numpy=True)

        except Exception as e:
            print("Exception caught. Skipping.")
            print(e)
            continue
            
        X2 = pd.DataFrame(train_embeddings)
        X2[target_col] = train_df[target_col]
        
        print("Writing to {}".format(new_fn))
        
        X2.to_csv(new_fn, index=False)
        


    