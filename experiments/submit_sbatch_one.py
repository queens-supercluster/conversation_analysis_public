#!/usr/bin/env python

import argparse
import subprocess
from itertools import product
import argparse
import os
import sys
import glob
import time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-r', '--rq', type=int, default=0)
parser.add_argument('-g', '--gig-mem', type=int, default=20)
parser.add_argument('-t', '--time-hours', type=int, default=48)
parser.add_argument('-v', '--venv-path', type=str, default="../ca_env5")
parser.add_argument('-c', '--copies', type=int, default=1)
#parser.add_argument('-p', '--project-dir', type = str, default="/global/project/hpcg1614_shared/ca/")
parser.add_argument('-p', '--project-dir', type = str, default="/global/home/hpc3552/conversation_analytics/experiments")
parser.add_argument('-o', '--overwrite-existing', type=str, default="no")
parser.add_argument('-n', '--n-bootstraps', type=int, default=30)
parser.add_argument('-d', '--dry-run', type=str, default="no")
args = parser.parse_args()

if args.rq == 0:
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
    
    
elif args.rq == 1:
    es = [
        # S-Bert, main models for text similarity task
        "all-distilroberta-v1", 
        "all-minilm-l12-v2", 
        'all-minilm-l6-v2', 
        "all-mpnet-base-v2", 
        "all-roberta-large-v1",

        # S-Bert, might be good, not sure
        "multi-qa-mpnet-base-dot-v1",
        "multi-qa-distilbert-cos-v1",

        # S-Bert, not very good at this task; ignore for now.
        "paraphrase-albert-small-v2",
        "paraphrase-minilm-l3-v2",
        #"allenai-specter",
        "facebook-dpr-ctx_encoder-multiset-base",
        "facebook-dpr-question_encoder-multiset-base",
        "msmarco-distilbert-cos-v5",
        #"distilbert-base-nli-stsb-quora-ranking",
        "msmarco-bert-co-condensor",
        "stsb-roberta-base-v2",
        #"nq-distilbert-base-v1",

        # Mean/Max Pooling (implemented from sentence-transformers package, but not S-BERT)
        #"albert-base-v2",
        #"t5-base",
        "average_word_embeddings_glove.6b.300d",
        "average_word_embeddings_komninos",
        "average_word_embeddings_levy_dependency",

        # Universal Sentence Encoder
        'universal-sentence-encoder',

        # OpenAI
        'text-similarity-ada-001',
        #'text-similarity-babbage-001',
        'text-similarity-curie-001',
        'text-similarity-davinci-001',
        
        'tf-idf',
    ]
    
    drs = ['umap']
    cs = ['birch', 'k-means', 'gmm']
    
elif args.rq == 2:
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
    cs = ['rf', 'lgbm', 'xgboost', 'lr']
    
elif args.rq == 3:
    #es = ["all-mpnet-base-v2"]
    es = ["all-roberta-large-v1"]
    es = ["tf-idf"]
    drs = ['none']
    cs = ['k-means']
    
elif args.rq == 4:
    es = ["tf-idf"]
    drs = ['none']
    cs = ['lda', 'nmf']
    
elif args.rq == 6:
    es = ["average_word_embeddings_glove.6b.300d"]
    drs = ['pca']
    cs = ['k-means']
    
    
already_done = 0
already_running = 0
need_to_run = 0
counter = 1

cmd = [ "squeue", "--user=hpc3552", "--format=\" %.80j\""]
result = subprocess.run(cmd, universal_newlines=True,  stdout=subprocess.PIPE)
sq_jobs = [n.replace("\"", '').strip() for n in result.stdout.splitlines() ]
    
datasets = ['banking77', 'clinc150', 'hwu64', ]
for d in datasets:
    for e in es:
        for dr in drs:
            for c in cs:


                #print("counter: {}".format(counter))
                counter = counter + 1
                name = "{}_{}_{}_{}_{}_search".format(d, 
                                                             e, 
                                                             dr, 
                                                             c, 
                                                             args.n_bootstraps, 
                                                 )

                out_dir = "{}/out/run_one".format(args.project_dir, name)
                df_fn = "{}/{}.csv".format(out_dir, name)

                if args.dry_run == "yes":
                    if os.path.isfile(df_fn):
                        #print("Already done: {}".format(df_fn))
                        already_done = already_done + 1
                    elif name in sq_jobs:
                        print("Already running: {}".format(df_fn))
                        already_running = already_running + 1
                    else:
                        print("\n\nNeed to run:  {}".format(df_fn))
                        need_to_run = need_to_run + 1
                        log_files = []
                        log_files = glob.glob("./logs/{}*.err".format(name))
                        for log_file in log_files:
                            print("From logfile {}".format(log_file))
                            os.system("tail {}".format(log_file))

                    continue

                # check if output file already exists
                if os.path.isfile(df_fn) and args.overwrite_existing == "no":
                    print("Warning: output file \"{}\" already exists. Skipping.".format(df_fn))
                    continue
                if name in sq_jobs and args.copies == 1:
                    print("Warning: Job {} appears to already be running. Skipping".format(name))
                    continue




                time_hours = args.time_hours
                if dr == "ica":
                    time_hours = time_hours * 2


                string = '\n'.join([
                    "#!/bin/bash",
                    "#SBATCH --job-name={}".format(name),
                    "#SBATCH --cpus-per-task=5",
                    "#SBATCH --account=def-hpcg1614",
                    "#SBATCH --qos=privileged",
                    "#SBATCH --mem={}gb".format(args.gig_mem),
                    "#SBATCH --time={}:00:00".format(args.time_hours),
                    "#SBATCH --output=logs/{}_%t_%j.out".format(name),
                    "#SBATCH --error=logs/{}_%t_%j.err".format(name),

                    "",
                    "pwd",
                    "module load StdEnv/2020 gcc/9.3.0 python/3.9",
                    "echo $PYTHONPATH",
                    "which python",
                    "python --version",
                    "source {}/bin/activate".format(args.venv_path),
                    "echo $PYTHONPATH",
                    "which python",
                    "python --version",
                    "ulimit -a",
                    "python run_one.py --dataset {} --embedding {} --dim-reduce {} --cluster {} ".format(d, e, dr, c),
                ]
                )

                print(string)

                f_name = 'tmp_scripts/tune_run_one.sh'
                with open(f_name, "w") as text_file:
                    text_file.write(string)
                    text_file.flush()
                    text_file.close()
                    
                time.sleep(10.0)    

                for i in range(args.copies):
                    cmd = [ "sbatch", f_name ]
                    p = subprocess.Popen(cmd, universal_newlines=True, bufsize=1)
                    exit_code = p.wait()
                    time.sleep(30.0)    
                    

if args.dry_run == "yes":
    print("Total Already done: {}".format(already_done))
    print("Total Already running: {}".format(already_running))
    print("Total Need to run: {}".format(need_to_run))
    print("Total: {}".format(already_done + already_running + need_to_run))
                        