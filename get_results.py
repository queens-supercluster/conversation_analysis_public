#!/usr/bin/env python

import pandas as pd
import glob
import argparse
import time
import sys
import numpy as np
from ast import literal_eval
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--project-dir', type = str, default="/global/home/hpc3552/conversation_analytics/experiments")
parser.add_argument('-o', '--out-dir', type = str, default="./results")
args = parser.parse_args()

experiments = glob.glob('{}/out/run_one/*.csv'.format(args.project_dir))
experiments = sorted(experiments)

res = []
all_dfs = []
best_trials = []

for fn in experiments:
    print(fn)
    
    try:
        _df = pd.read_csv(fn)
        all_dfs.append(_df)
        
    except Exception as e:
        print("Warning in read_csv: {}".format(e), file=sys.stderr)
        sys.exit(1)
        #continue
        
    try:
        list_cols = [ 
            'user_attrs_boostrap_ids',
            'user_attrs_cluster_n_unique',
            'user_attrs_cluster_n_neg',
            'user_attrs_duration_sec',
            'user_attrs_ARI',
            'user_attrs_AMI',
            'user_attrs_Completeness',
            'user_attrs_Fowlkes_Mallows',
            'user_attrs_Homogeneity',
            'user_attrs_V_Measure',
        ]
        for col in list_cols:
            _df[col] = _df[col].apply(lambda x: [] if x is np.nan else literal_eval(x))
    except Exception as e:
        print("Warning in literal_eval: {}".format(e), file=sys.stderr)
        print(_df.tail(3).T)
        sys.exit(1)
        #continue
       
    # Drop columns that start with "param_" (will be different for each experiment)
    _df = _df[_df.columns.drop(list(_df.filter(regex='params_')))]
        
    _df = _df[_df['state'] == "COMPLETE"]
    _df = _df.set_index(['number']).apply(pd.Series.explode).reset_index()
    _df = _df.sort_values(['value', 'user_attrs_boostrap_ids'], ascending=[False, True])
    
    best_trial = _df['number'].iloc[0]
    best_trials.append(_df[_df['number']==best_trial])
    
best_df = pd.concat(best_trials)
timestr = time.strftime("%Y%m%d-%H%M%S")
best_df.to_csv("{}/best_results_{}.csv".format(args.out_dir, timestr), index=False)

scores_df =  best_df.groupby(['user_attrs_dataset', 'user_attrs_embedding', 'user_attrs_dim_reduce', 'user_attrs_cluster']).agg(
        n_boostraps=('user_attrs_boostrap_ids', 'count'),
        mean_AMI=('user_attrs_AMI', 'mean'),
        mean_duration_sec=('user_attrs_duration_sec', 'mean'),
    ).sort_values(['user_attrs_dataset', 'mean_AMI'], ascending=False).reset_index()

scores_df.to_csv("{}/scores_{}.csv".format(args.out_dir, timestr), index=False)
        

