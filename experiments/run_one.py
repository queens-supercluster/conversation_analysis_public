import argparse
import pandas as pd
import os
import sys
import time
import numpy as np

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# To be able to get __version__
import sklearn, umap, lightgbm, xgboost

from sklearn.feature_extraction.text import TfidfVectorizer

from umap import UMAP
from sklearn.decomposition import PCA, TruncatedSVD, FastICA

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.cluster import Birch, AgglomerativeClustering, HDBSCAN
from sklearn.cluster import KMeans

from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.metrics.cluster import adjusted_mutual_info_score, completeness_score, fowlkes_mallows_score, adjusted_rand_score, homogeneity_score, v_measure_score
from sklearn.utils import resample

from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.decomposition import NMF, LatentDirichletAllocation

from scipy.spatial import distance

import optuna

###############################################################################
# Command line args
###############################################################################
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset', type=str, default="banking77")
parser.add_argument('-e', '--embedding', type=str, default="all-mpnet-base-v2")
parser.add_argument('-r', '--dim-reduce', type=str, default="umap")
parser.add_argument('-c', '--cluster', type=str, default="birch")
parser.add_argument('-n', '--n-bootstraps', type=int, default=30)
parser.add_argument('-o', '--overwrite-existing', type=str, default="no")
#parser.add_argument('-p', '--project-dir', type = str, default="/global/project/hpcg1614_shared/ca/")
parser.add_argument('-p', '--project-dir', type = str, default="/global/home/hpc3552/conversation_analytics/experiments")
parser.add_argument('-t', '--test-run', type=str, default="no")

args = parser.parse_args()

# Some sanity checks

if (args.cluster.lower() == "lda" or args.cluster.lower() == "nmf") and (args.dim_reduce.lower() != "none" or args.embedding.lower() != "tf-idf"):
    print("Error: Unsupported combination: {}, {}, {}.".format(args.embedding, args.dim_reduce, args.cluster), file=sys.stderr)
    sys.exit(1)
    
random_state = 42

# Names for output files
timestr = time.strftime("%Y%m%d-%H%M%S")

experiment_name = "{}_{}_{}_{}_{}_search".format(args.dataset, args.embedding, args.dim_reduce, args.cluster, args.n_bootstraps)

# Make sure the output directory has been created
out_dir = "{}/out/run_one".format(args.project_dir, experiment_name)
df_fn = "{}/{}.csv".format(out_dir, experiment_name)

# check if output file already exists
if os.path.isfile(df_fn) and args.overwrite_existing == "no":
    print("Warning: output file \"{}\" already exists. Skipping.".format(df_fn))
    sys.exit(0)

if args.dataset == "banking77":
    train_dir = "{}/data/banking77/".format(args.project_dir)
    text_col = "text"
    target_col = "category"
elif args.dataset == "amazon_all":
    train_dir = "{}/data/amazon-all".format(args.project_dir)
    text_col = "utterance"
    target_col = "intent"
elif args.dataset == "clinc150":
    train_dir = "{}/data/clinc150".format(args.project_dir)
    text_col = "text"
    target_col = "label"
elif args.dataset == "hwu64":
    train_dir = "{}/data/hwu64".format(args.project_dir)
    text_col = "text"
    target_col = "label"
else:
    print("Error: Unknown dataset \"{}\"".format(args.dataset), file=sys.stderr)
    sys.exit(1)


print("Running experiment: {}".format(experiment_name))


def objective_func(trial, args, train_dir, text_col, target_col):
   
    # Containers to hold these scores per bootstrap
    durations = []
    bootstraps = []
    ARIs = []
    AMIs = []
    completenesses = []
    fws = []
    homogeneities = []
    v_measures = []
    n_uniques = []
    n_negs = []
    
    
    ###############################################################################
    # Get Embeddings
    ###############################################################################
    # Assumes embeddings have already been computed

    if args.embedding == "tf-idf":
        new_fn = "{}/clean.csv".format(train_dir)

        train_df = pd.read_csv(new_fn)
        X = train_df[text_col]
        y = train_df[target_col]

        print("\nCreating TF-IDF...")

        vectorizer = TfidfVectorizer(
            min_df=1,
            lowercase=True,
            ngram_range=(1,2),
            stop_words = "english",
            analyzer="word",
            max_features=500,
            max_df=1.0,
        )

        _X_embeddings = vectorizer.fit_transform(X)

         # Turn from sparse to dense (because ICA and PCA complain otherwise)
        _X_embeddings = _X_embeddings.toarray()

        _y = y

    else:
        print("\nLoading embeddings...")

        # Special workaround. Some files on disk have capital letters, but the name of the
        # embedding needs to be lowercase. We'll map them out here manually.
        special_file_names = {
            'all-minilm-l12-v2': 'all-MiniLM-L12-v2',
            'all-MiniLM-l6-v2': 'all-MiniLM-L6-v2', 
            'paraphrase-minilm-l3-v2': 'paraphrase-MiniLM-L3-v2',
            'average_word_embeddings_glove.6b.300d': 'average_word_embeddings_glove.6B.300d', 
        }

        file_base = special_file_names.get(args.embedding, args.embedding)
        new_fn = "{}/clean_embed_{}.csv".format(train_dir, file_base)

        if not (os.path.isfile(new_fn)):
            print("File {} does not exist. Exiting.".format(new_fn), file=sys.stderr)
            sys.exit(1)

        train_df = pd.read_csv(new_fn)

        _X_embeddings = train_df.drop([target_col], axis=1).to_numpy()
        _y = train_df[target_col]
        
    # Get/Create embeddings (outside loop to avoid unnecessary duplication of work)
    dim_reducer_params = {}
    dim_reducer = None
    if args.dim_reduce.lower() == "umap":
        dim_reducer_params = {
            'n_components': 20,
            'n_neighbors': 18,
            'min_dist': 0.1,
            'metric': "euclidean",
            'init': "spectral",
            'learning_rate': 1,
            'n_epochs': None,
            'spread': 1,
            'low_memory': False,
            'set_op_mix_ratio': 1,
            'local_connectivity': 1,
            'random_state': 42,
            'n_jobs': 1, # must be 1 when random_state is set
        }
        dim_reducer = UMAP(**dim_reducer_params)

    elif args.dim_reduce.lower() == "pca":
        dim_reducer_params = {
            'n_components': 14,
            'svd_solver': "full",
            'random_state': 42,
        }
        dim_reducer = PCA(**dim_reducer_params)
    elif args.dim_reduce.lower() == "ica":
        dim_reducer_params = {
            'n_components': 12,
            'algorithm': "parallel",
            'fun': 'exp',
            'max_iter': 300,
            'whiten': 'unit-variance',
            'random_state': 42,
        }
        dim_reducer = FastICA(**dim_reducer_params)
    elif args.dim_reduce.lower() == "svd":
        dim_reducer_params = {
            'n_components': 12,
            'algorithm': "randomized",
            'n_iter': 5,
            'random_state': 42,
        }
        dim_reducer = TruncatedSVD(**dim_reducer_params)
    elif args.dim_reduce.lower() == "none":
        pass
    else:
        print("Error: unknown dim reducer: {}. Exiting".format(args.dim_reduce), file=sys.stderr)
        sys.exit(1)

    ###############################################################################
    # Dimensionality Reduction
    ###############################################################################
    print("\nDim reduction...")

    if dim_reducer is None:
        # Do nothing!
        _X_dims = _X_embeddings
    else:
        try:
            print("Fitting dim reducer:")
            print(dim_reducer)
            _X_dims = dim_reducer.fit_transform(_X_embeddings)
        except Exception as e:
            print("Caught exception during DR fit.", file=sys.stderr)
            print(e, file=sys.stderr)
            print("Pruning", file=sys.stderr)
            raise optuna.TrialPruned()

    print("\nX dim head and shape:")
    print(_X_dims[:5,:])
    print(_X_dims.shape)

    # In certain rare cases, UMAP was returning some NANs, and K-means crashed.
    # I couldn't quite figure out why UMAP was doing so (it might have to do with
    # disconnected points), so, given it was so isolated and rare). 
    print(np.any(np.isnan(_X_dims)))
    print(np.where(np.isnan(_X_dims)))
    _X_dims = np.nan_to_num(_X_dims)
    n_rows, n_cols = _X_dims.shape
    
    
    ##################################################
    # Build cluster object 
    ##################################################

    clusterer_params = {}
    clusterer = None
    #n_samples = None # For boostrapping; None means "all"
    n_samples = int(n_rows * 0.25) # Tmp; make an input to this script.

    if args.cluster.lower() == "birch":
        clusterer_params = {
            "n_clusters": trial.suggest_int("n_clusters", 2, 1000),
            "branching_factor": trial.suggest_int("branching_factor", 1, 1000),
            "threshold": trial.suggest_float("threshold", 0.0, 1.0),
        }
        clusterer = Birch(**clusterer_params)
    elif args.cluster.lower() == "k-means":
        clusterer_params = {
            "n_clusters": trial.suggest_int("n_clusters", 2, 1000),

            "init": "k-means++",
            "n_init": 'auto',
            "max_iter": 1000,
            "algorithm": "lloyd",
            "random_state": 0,
        }
        clusterer = KMeans(**clusterer_params)
    elif args.cluster.lower() == "hierarchical":
        clusterer_params = {
            "n_clusters": trial.suggest_int("n_clusters", 2, 1000),
            "linkage": trial.suggest_categorical("linkage", [ 'ward', 'complete', 'average', ]),
        }
        clusterer = AgglomerativeClustering(**clusterer_params)
    elif args.cluster.lower() == "hdbscan":
        clusterer_params = {
            "min_cluster_size": trial.suggest_int("min_cluster_size", 1, 1000),
            "min_samples": trial.suggest_int("min_samples", 1, 1000),
            #"metric": trial.suggest_categorical("metric", ['euclidean', 'cosine']),
            "metric": "precomputed",
            #"cluster_selection_method": trial.suggest_categorical("cluster_selection_method", ['eom']),
            "cluster_selection_method": 'eom',
            #"cluster_selection_epsilon": trial.suggest_float("cluster_selection_epsilon", 0, 1000),
            "cluster_selection_epsilon": 0.0,
            "n_jobs": 3,
        }
        clusterer = HDBSCAN(**clusterer_params)
        
    elif args.cluster.lower() == "lda":
        clusterer_params = {
            "n_components": trial.suggest_int("n_clusters", 2, 1000),
            "doc_topic_prior": trial.suggest_float("doc_topic_prior", 0.00, 1.00),
            "topic_word_prior": trial.suggest_float("topic_word_prior", 0.00, 1.00),

            "learning_method": 'batch',
            "max_iter": 50,
            "random_state": 42,
            "n_jobs": 3,
        }
        clusterer = LatentDirichletAllocation(**clusterer_params)
    elif args.cluster.lower() == "nmf":
        clusterer_params = {
            "n_components": trial.suggest_int("n_clusters", 2, 1000),
            "alpha_W": trial.suggest_float("alpha_W", 0.00, 1.00),
            "alpha_H": trial.suggest_float("alpha_H", 0.00, 1.00),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.00, 1.00),

            # Note: due to an odd bug deep in scipy, that is only tickled on slurm batch jobs, leave init as random for now
            # Spent 4 hours figuring this one out!
            "init": 'random',
            "max_iter": 500,
            "solver": "mu",
            "random_state": 0,
        }
        clusterer = NMF(**clusterer_params)
    elif args.cluster.lower() == "rf":
        clusterer_params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
            "max_depth": trial.suggest_int("max_depth", 1, 1000),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 1000),
            
            "random_state": 42,
        }
        clusterer = RandomForestClassifier(**clusterer_params)
    elif args.cluster.lower() == "lgbm":
        clusterer_params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 10000),
            "learning_rate": trial.suggest_float("learning_rate", 0.0, 100.0),
            "num_leaves": trial.suggest_int("num_leaves", 1, 1000),
            
            "max_depth": 8,
            "min_split_gain": 0.01,
            "colsample_bytree": 0.6,
            "n_jobs": 3,
            "subsample": 0.75,
            "subsample_freq":10, 
            
            "random_state": 42,
        }
        clusterer = LGBMClassifier(**clusterer_params)
    elif args.cluster.lower() == "xgboost":
        clusterer_params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 10000),
            "learning_rate": trial.suggest_float("learning_rate", 0.0, 100.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 1000),
            
            "tree_method": "hist",
            "random_state": 42,
        }
        clusterer = XGBClassifier(**clusterer_params)
    elif args.cluster.lower() == "knn":
        clusterer_params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 1000),
            
            "weights": "uniform",
            "metric": "precomputed",
            "n_jobs": 3,
        }
        clusterer = KNeighborsClassifier(**clusterer_params)
    elif args.cluster.lower() == "lr":
        clusterer_params = {
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "C": trial.suggest_float("C", 0.01, 10.0),
            
            "solver": "saga",
            "penalty": "elasticnet",
            "random_state": 42,
            "n_jobs": 3,
        }
        clusterer = LogisticRegression(**clusterer_params)
    elif args.cluster.lower() == "histgbc":
        clusterer_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.0, 100.0),
            
            "max_iter": 100,
            "max_depth": 8,
            "max_bins": 127, 
            "early_stopping": True, 
        }
        clusterer = HistGradientBoostingClassifier(**clusterer_params)
    else:
        print("Error: unknown clusterer: {}. Exiting".format(args.cluster), file=sys.stderr)
        sys.exit(1)


    n_bootstraps = args.n_bootstraps
    if args.cluster.lower() in ['rf', 'lgbm', 'xgboost', 'knn', 'lr', 'histgbc']:
        n_bootstraps = 5
    sum_AMI_so_far = 0.
    for i in range(0, n_bootstraps):

        start = time.time()
        print("\n Running bootstrap {} of {}".format(i, n_bootstraps))

        X_dims, y = resample(
            _X_dims, _y, 
            replace=True, 
            n_samples=n_samples,
            stratify=_y,
            random_state=random_state+i)
        
        print("X_dims.shape: {}".format(X_dims.shape))

        ###############################################################################
        # Clustering
        ###############################################################################
        
        # The following algorithms can have precomputed distance metrics; do so now.
        if args.cluster.lower() in ['hdbscan']:
            print("Precomputing distance matrix.")
            X_dims = distance.cdist(X_dims, X_dims, trial.suggest_categorical("metric", ['euclidean', 'cosine']))
            print("X_dims.shape: {}".format(X_dims.shape))
            
        if args.cluster.lower() == "nmf":

            try:
                print("Fitting NMF:")
                print(clusterer)
                W = clusterer.fit_transform(X_dims)
                W = np.around(normalize(W, axis=1, norm='l1'), decimals=4)
                X_topic_ids = np.argmax(W, axis=1)
            except Exception as e:
                print("Caught exception during NMF fit.", file=sys.stderr)
                print(e, file=sys.stderr)
                print("Pruning", file=sys.stderr)
                raise optuna.TrialPruned()

        elif args.cluster.lower() == "lda":
            try:
                print("Fitting LDA:")
                print(clusterer)
                # Theta = document-topic matrix
                theta = clusterer.fit_transform(X_dims)
                X_topic_ids = np.argmax(theta, axis=1)
            except Exception as e:
                print("Caught exception during LDA fit.", file=sys.stderr)
                print(e, file=sys.stderr)
                print("Pruning", file=sys.stderr)
                raise optuna.TrialPruned()
                
        elif args.cluster.lower() in ['rf', 'lgbm', 'xgboost', 'knn', 'lr', 'histgbc']:
            try:
                print("Fitting classifier:")
                print(clusterer)
                if args.cluster.lower() == "lgbm":
                    print(clusterer_params)
                    print(clusterer.get_params())
                
                X_train, X_test, y_train, y_test = train_test_split(X_dims, y, test_size=0.20, random_state=42)
               
                if args.cluster.lower() in ['knn']:
                    print("Precomputing distance matrix.")

                    scaler = MinMaxScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    X_train = distance.cdist(X_train, X_train, trial.suggest_categorical("metric", ['euclidean', 'cosine']))
                    print("X_train.shape: {}".format(X_train.shape))
                    
                clusterer.fit(X_train, y_train)
                X_cluster_ids = clusterer.predict(X_test)
                X_topic_ids = X_cluster_ids
               
                # For the purposes of metric calculation below
                y = y_test
            except Exception as e:
                print("Caught exception during classifier fit.", file=sys.stderr)
                print(e, file=sys.stderr)
                print("Pruning", file=sys.stderr)
                raise optuna.TrialPruned()

        else:

            try:
                print("Fitting clusterer:")
                    
                print(clusterer)
                X_cluster_ids = clusterer.fit_predict(X_dims)
                X_topic_ids = X_cluster_ids
            except Exception as e:
                print("Caught exception during cluster fit.", file=sys.stderr)
                print(e, file=sys.stderr)
                print("Pruning", file=sys.stderr)
                raise optuna.TrialPruned()

        print("Topic IDs:")
        print(type(X_topic_ids))
        n_unique = len(np.unique(X_topic_ids))
        n_neg = int((X_topic_ids==-1).sum())
        print(n_unique)
        print(n_neg)


        ###############################################################################
        # Calculate Metrics on Topic Assignments
        ###############################################################################
        # Adjusted rand index is the main metric we care about, but we will measure the other available
        # metrics in sklearn because they may be useful for further analysis.

        adjusted_rand = adjusted_rand_score(y, X_topic_ids)
        adjusted_mutual_info = adjusted_mutual_info_score(y, X_topic_ids)
        completeness = completeness_score(y, X_topic_ids)
        fowlkes_mallows = fowlkes_mallows_score(y, X_topic_ids)
        homogeneity = homogeneity_score(y, X_topic_ids)
        v_measure = v_measure_score(y, X_topic_ids)

        end = time.time()
        duration = end - start
        
        durations.append(duration)
        bootstraps.append(i)
        n_uniques.append(n_unique)
        n_negs.append(n_neg)
        ARIs.append(adjusted_rand)
        AMIs.append(adjusted_mutual_info)
        completenesses.append(completeness)
        fws.append(fowlkes_mallows)
        homogeneities.append(homogeneity)
        v_measures.append(v_measure)

        print("Duration: {:4f}".format(duration))
        print("AMI: {:4f}".format(adjusted_mutual_info))

        # Handle pruning based on the intermediate value.
        sum_AMI_so_far = sum_AMI_so_far + adjusted_mutual_info
        mean_AMI_so_far = sum_AMI_so_far / (i+1)
        trial.report(mean_AMI_so_far, i)
        if trial.should_prune():
            print("Pruning trial early with low score: {}".format(mean_AMI_so_far))
            raise optuna.TrialPruned()
            
    trial.set_user_attr('timestr', timestr)
    trial.set_user_attr('trial', trial.number)
    trial.set_user_attr('sklearn_version', sklearn.__version__)
    trial.set_user_attr('umap_version', umap.__version__)
    trial.set_user_attr('numpy_version', np.__version__)
    trial.set_user_attr('lightgbm_version', lightgbm.__version__)
    trial.set_user_attr('xgboost_version', xgboost.__version__)
    trial.set_user_attr('dataset', args.dataset)
    trial.set_user_attr('use_best', "search")
    trial.set_user_attr('embedding', args.embedding)
    trial.set_user_attr('dim_reduce', args.dim_reduce)
    trial.set_user_attr('dim_reduce_repr', dim_reducer.__repr__())
    trial.set_user_attr('cluster', args.cluster)
    trial.set_user_attr('cluster_repr', clusterer.__repr__())
    trial.set_user_attr('boostrap_ids', bootstraps)
    trial.set_user_attr('cluster_n_unique', n_uniques)
    trial.set_user_attr('cluster_n_neg', n_negs)
    trial.set_user_attr('duration_sec', durations)
    trial.set_user_attr('ARI', ARIs)
    trial.set_user_attr('AMI', AMIs)
    trial.set_user_attr('Completeness', completenesses)
    trial.set_user_attr('Fowlkes_Mallows', fws)
    trial.set_user_attr('Homogeneity', homogeneities)
    trial.set_user_attr('V_Measure', v_measures)
        
       
    # Return average AMI across the bootstraps
    return sum_AMI_so_far / n_bootstraps

        
search_space = {}
        
if args.cluster.lower() == "birch":
    search_space['threshold'] = [float(i) for i in np.linspace(0.01, 1.0, num=4, dtype=float)]
    search_space['branching_factor'] = [int(i) for i in np.arange(25, 125, step=25, dtype=int)]
elif args.cluster.lower() == "k-means":
    pass
elif args.cluster.lower() == "hierarchical":
    search_space['linkage'] = ['ward', 'complete', 'average', ]
elif args.cluster.lower() == "hdbscan":
    search_space['min_cluster_size'] = [int(i) for i in np.arange(2, 30, step=5, dtype=int)]
    search_space['min_samples'] = [int(i) for i in np.arange(2, 30, step=5, dtype=int)]
    search_space['metric'] = ['euclidean', 'cosine']
    #search_space["cluster_selection_method"] = ['eom']
    #search_space['cluster_selection_epsilon'] = [float(i) for i in np.linspace(0.0, 100.0, num=8, dtype=float)]
elif args.cluster.lower() == "lda":
    search_space['doc_topic_prior'] = [float(i) for i in np.linspace(0.0, 0.03, num=4, dtype=float)]
    search_space['topic_word_prior'] = [float(i) for i in np.linspace(0.0, 0.03, num=4, dtype=float)]
elif args.cluster.lower() == "nmf":
    search_space["alpha_W"] = [0.0, 0.1]
    search_space["alpha_H"] = [0.0]
    search_space["l1_ratio"] = [0.0, 0.1]
elif args.cluster.lower() == "rf":
    search_space['n_estimators'] = [10, 100, 500]
    search_space['max_depth'] = [5, 20, 100]
    search_space['min_samples_leaf'] = [1, 10, 20]
elif args.cluster.lower() == "lgbm":
    search_space['n_estimators'] = [10, 100, 500, 1000]
    search_space['learning_rate'] = [0.01, 0.03, 0.1, 0.2, 0.3]
    search_space['num_leaves'] = [15, 31, 127]
elif args.cluster.lower() == "xgboost":
    search_space['n_estimators'] = [10, 100, 500]
    search_space['learning_rate'] = [0.01, 0.1, 0.3]
    search_space['min_child_weight'] = [1, 5, 10]
elif args.cluster.lower() == "knn":
    search_space['n_neighbors'] = [5, 15, 31]
    
    # Note: "cosine" caused negative values in precomputed dist matrix, which causes
    # KNN to complain
    search_space['metric'] = ['euclidean']
elif args.cluster.lower() == "lr":
    search_space['l1_ratio'] = [0.0, 0.5, 1.0]
    search_space['C'] = [0.05, 1.0, 3.0]
elif args.cluster.lower() == "histgbc":
    search_space['learning_rate'] = [0.01, 0.1, 0.3]


# The search space for n_clusters (n_components for nmf and lda) depends on the dataset
if args.cluster.lower() not in ['hdbscan', 'rf', 'lgbm', 'xgboost', 'knn', 'lr', 'histgbc']:
    if args.dataset == "banking77":
        search_space['n_clusters'] = [int(i) for i in np.arange(50, 90, step=10, dtype=int)]
    elif args.dataset == "amazon_all":
        search_space['n_clusters'] = [int(i) for i in np.arange(20, 80, step=10, dtype=int)]
    elif args.dataset == "clinc150":
        search_space['n_clusters'] = [int(i) for i in np.arange(130, 160, step=10, dtype=int)]
    elif args.dataset == "hwu64":
        search_space['n_clusters'] = [int(i) for i in np.arange(50, 75, step=5, dtype=int)]
           
n_trials = None
if args.test_run == "yes":
    print("Warning: Test run only")
    n_trials = 2

print("Grid size:")
size=1
for _key in search_space:
    size = size * len(search_space[_key])
    print("Key={}, len={}, new_size={}".format(_key, len(search_space[_key]), size))
print("Total grid size: {}".format(size))

def print_best_callback(study, trial):
    try:
        print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    except Exception as e:
        print("Caught exception while printing best trial. Skipping.", file=sys.stderr)

sampler = optuna.samplers.GridSampler(search_space)

pruner = optuna.pruners.PercentilePruner(
    10.0, 
    n_startup_trials=3,
    n_warmup_steps=0, 
    interval_steps=1)
study_name = "{}_{}".format(experiment_name, target_col)
storage = "sqlite:///{}/{}.db".format(out_dir, experiment_name)
        
study = optuna.create_study(
    study_name=experiment_name, 
    sampler=sampler, 
    pruner=pruner, 
    storage = storage,
    load_if_exists=True,
    direction="maximize")

study.optimize(lambda trial: objective_func(trial, args, train_dir, text_col, target_col),
               n_trials=n_trials,
               gc_after_trial=True,
               callbacks=[print_best_callback]
              )


res_df = study.trials_dataframe()
print(res_df)
if args.test_run != "yes":
    print("Writing output file:  {}".format(df_fn))
    res_df.to_csv(df_fn, index=False)