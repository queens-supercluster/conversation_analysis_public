# Experiments

This directory holds scripts to design, execute, and analyze the experiments for this project.


## Study Design 

Conceptually, the study has three main experimental factors:

1. Embedding technique/model. There are about 20 options we will consider, such as:
  - SBERT (Up to 17 options)
  - Pooling (4 options), 
  - USE (1 level)
  - TF-IDF
  - OpenAI GPT-3 (2-3 options)
  - ... and possibly more in the future.
  
2. Dimensionality reduction technique. ~ 4 options, including:
  - None, UMAP, SVD, PCA, possibly more...
  
3. Cluster Algorithm: ~ 6 options, including:
  - K-means, Hierachical, GMM, DBSCAN, HDBSCAN, 
  - LDA, NMF
  - RF, XGBoost, LGBM, LR
  
Each combination of factor levels is dubbed an "experiment" or a "treatment."
  
(Note, for the purposes of this study, we are calling statistical topic models like LDA and NMF clustering
algorithms, even though technically they are not. Same with supervised classification, like RF.)
 
## Experiment Script

`run_one.py` script will run one combination above on one dataset. E.g., running


```
python run_one.py -d clinc150 -e text-similarity-davinci-001 -r ica -c lgbm
```

will run LGBM on Davinci embeddings with ICA dim reduction on the clinc150 dataset.

`run_one.py` uses Optuna to run a grid hyperparameter search on the clustering algorithm, and will output
a `.csv` file containing the results (metrics, runtime, environment details) of each Optuna trial.

Run 

```
python run_one.py --help
``` 

for a list of all command line options.


## Directory Structure

The data for the experiments, and the output directory, are assumed to be in a common directory, called the project directory. The project directory can be set via the `--project-dir` command line option.

## Using SLURM

To run many experiments on CAC's SLURM cluster in parallel, run:

```
./submit_sbatch_one.py --rq 0
```

The script will submit a batch job for many treatments (i.e., for all embedding models) on all datasets.

The `--rq 0` command line parameter defines a set of runs to submit; see and modify the script for details.

The `--dry-run yes` command line parameter will monitor the status of runs: how many have completed, how many are running, and how many still need to be run.

See `--help` for more details.

## Analyzing Results

Use the `get_results.py` script to analyze all the output `.csv` files and create two summary files, which are suitable for further analysis.

