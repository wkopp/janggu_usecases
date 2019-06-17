from itertools import product
import argparse
import os
import numpy as np
import matplotlib
import pandas as pd
from model_utils import objective
matplotlib.use('Agg')


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"



np.random.seed(1234)

# Fetch parser arguments
PARSER = argparse.ArgumentParser(description='DNA model.')

PARSER.add_argument('-inputpath', dest='inpath',
                    default='../data', help='Location of input files')

PARSER.add_argument('-path', dest='path',
                    default='../jund_results',
                    help="Output directory for the examples.")

args = PARSER.parse_args()

os.environ['JANGGU_OUTPUT'] = args.path

inpath = args.inpath

# load the dataset

# first do an exhaustive grid search

print("#" * 20)
print("Test effect of scanning single or both strands and higher-order motifs")


results = {'auprc_val':[], 'auprc_test':[], 'dropout':[], 'order':[], 'strand':[], 'modelname':[]}

run = 5
for order, sdrop, rep in product([3, 2, 1], [0.0, 0.2], [1, 2, 3, 4, 5]):
    shared_space = {'type': 'dna_only',
                    'name': 'dna_o{}_d{}_run_{}_{}'.format(order, sdrop, run, rep),
                    'binsize': 200,
                    'epochs': 30,
                    'seq_dropout': sdrop,
                    'dnaflank': 150,
                    'dnaseflank': 0,
                    'order': order,
                    'stranded': 'double',
                    'nmotifs1': 10,
                    'motiflen': 11,
                    'pool1': 30,
                    'stride': 1,
                    'shift_range': 0,
                    'nmotifs2': 8,
                    'hypermotiflen': 3,
                    'opt': 'amsgrad'}

    print(shared_space['name'])
    res = objective(shared_space)
    results['auprc_val'].append(res['auprc_val'])
    results['auprc_test'].append(res['auprc_test'])
    results['dropout'].append(sdrop)
    results['order'].append(order)
    results['strand'].append(shared_space['stranded'])
    results['modelname'].append(res['modelname'])
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.environ['JANGGU_OUTPUT'],
                           "dna_gridsearch_{}.tsv".format(run)), sep='\t')
