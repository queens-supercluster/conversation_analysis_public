import numpy as np
import pandas as pd
import os
import sys
from numpy import count_nonzero
from pathlib import Path


res = []

es = [
    "all-distilroberta-v1", 
    "all-minilm-l12-v2", 
    "all-mpnet-base-v2", 
    "all-roberta-large-v1",

    'text-similarity-ada-001',
    'text-similarity-curie-001',
    'text-similarity-davinci-001',

    "average_word_embeddings_glove.6b.300d",
    "average_word_embeddings_komninos",
    "average_word_embeddings_levy_dependency",

    'universal-sentence-encoder',
]
ds = ['banking77', 'clinc150', 'hwu64', ]

# Special workaround. Some files on disk have capital letters, but the name of the
# embedding needs to be lowercase. We'll map them out here manually.
special_file_names = {
    'all-minilm-l12-v2': 'all-MiniLM-L12-v2',
    'all-MiniLM-l6-v2': 'all-MiniLM-L6-v2', 
    'paraphrase-minilm-l3-v2': 'paraphrase-MiniLM-L3-v2',
    'average_word_embeddings_glove.6b.300d': 'average_word_embeddings_glove.6B.300d', 
}

project_dir = '/global/home/hpc3552/conversation_analytics/experiments'

for d in ds:
    for e in es:
        print("Dataset: {}, Embeddings: {}".format(d, e))
        
        if d == "banking77":
            train_dir = "{}/data/banking77/".format(project_dir)
            text_col = "text"
            target_col = "category"
        elif d == "clinc150":
            train_dir = "{}/data/clinc150".format(project_dir)
            text_col = "text"
            target_col = "label"
        elif d == "hwu64":
            train_dir = "{}/data/hwu64".format(project_dir)
            text_col = "text"
            target_col = "label"
        else:
            print("Error: Unknown dataset \"{}\"".format(d), file=sys.stderr)
            sys.exit(1)

        

        file_base = special_file_names.get(e, e)
        new_fn = "{}/clean_embed_{}.csv".format(train_dir, file_base)

        if not (os.path.isfile(new_fn)):
            print("File {} does not exist. Exiting.".format(new_fn), file=sys.stderr)
            sys.exit(1)

        train_df = pd.read_csv(new_fn)

        _X_embeddings = train_df.drop([target_col], axis=1).to_numpy()
        _y = train_df[target_col]


        rows = _X_embeddings.shape[0]
        cols = _X_embeddings.shape[1]
        nz = count_nonzero(_X_embeddings)
        size = float(_X_embeddings.size)
        sparsity = 1.0 - ( count_nonzero(_X_embeddings) / float(_X_embeddings.size))

        res.append(
            {
              "Dataset": d,
              "Embedding": e,
              "Rows": rows,
              "Cols": cols,
              "Nonzero": nz,
              "Size": size,
              "Sparsity": sparsity,
            }
        )
    
    
datasets = pd.DataFrame(res)

print("\n# Datasets Summary")
print(datasets.to_markdown())