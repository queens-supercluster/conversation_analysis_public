#!/usr/bin/env python

import subprocess
import argparse
import os
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-r', '--run-all', type=int, default=0)
parser.add_argument('-p', '--project-dir', type = str, default="./")
parser.add_argument('-o', '--overwrite-existing', type=str, default="no")
args = parser.parse_args()

if args.run_all == 1:
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
        'tf-idf',
    ]
    drs = ['umap', 'ica', 'none', 'pca', 'svd' ]
    cs = ['k-means', 'birch', 'hierarchical', 'hdbscan']
    datasets = ['banking77', 'clinc150', 'hwu64', ]
else:
    es = [ "all-mpnet-base-v2",  'tf-idf', ]
    drs = ['umap']
    cs = ['birch']
    datasets = ['banking77']
    
for d in datasets:
    for e in es:
        for dr in drs:
            for c in cs:

                name = "{}_{}_{}_{}_search".format(d, e, dr, c,)

                out_dir = "{}/results/run_one".format(args.project_dir, name)
                df_fn = "{}/{}.csv".format(out_dir, name)

                # check if output file already exists
                if os.path.isfile(df_fn) and args.overwrite_existing == "no":
                    print("Warning: output file \"{}\" already exists. Skipping.".format(df_fn))
                    continue
                    
                cmd = [ "python", "run_one.py",  
                        "--dataset", d,
                        "--embedding", e, 
                        "--dim-reduce", dr,
                        "--cluster", c,
                       ]
                print(cmd)
                p = subprocess.Popen(cmd, universal_newlines=True, bufsize=1)
                exit_code = p.wait()