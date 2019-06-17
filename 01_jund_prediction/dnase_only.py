import argparse
import os
import matplotlib
import pandas as pd
from itertools import product
matplotlib.use('Agg')

import numpy as np

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model_utils import objective

np.random.seed(1234)

# Fetch parser arguments
PARSER = argparse.ArgumentParser(description='Command description.')

PARSER.add_argument('-inputpath', dest='inpath',
                    default='../data', help='Location of input files')

PARSER.add_argument('-path', dest='path',
                    default='../jund_results',
                    help="Output directory for the examples.")

args = PARSER.parse_args()

os.environ['JANGGU_OUTPUT'] = args.path

inpath = args.inpath

# load the dataset

print("#" * 20)
print("Test effect of DNASE on BAM files (region width and normalization)")

results = {'auprc_val':[], 'auprc_test': [], 'normalize':[], 'augment': [], 'modelname': []}

run = 7
for normalize, augment, rep in product(['tpm', 'zscorelog', 'zscore', None],
                                       ['orient', 'none'], [1, 2, 3, 4, 5]):
    shared_space = {
        'type': 'dnase_bam_only',
        'name': 'dnase_run_{}_{}_{}_{}'.format(normalize, augment, run, rep),
        'binsize': 200,
        'epochs': 30,
        'dnaseflank': 450,
        'normalize': normalize,
        'augment': augment,
        'nkernel1': 10,
        'kernel1len': 5,
        'kernel1pool': 2,
        'nkernel2': 5,
        'kernel2len': 3,
        'opt': 'amsgrad'}
    print(shared_space['name'])
    res = objective(shared_space)
    results['auprc_val'].append(res['auprc_val'])
    results['auprc_test'].append(res['auprc_test'])
    results['modelname'].append(res['modelname'])
    results['normalize'].append(normalize if normalize is not None else 'none')
    results['augment'].append(augment)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.environ['JANGGU_OUTPUT'],
                           "dnase_gridsearch_{}.tsv".format(run)), sep='\t')


print("#" * 20)

