import argparse
import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = ' 0'


from janggu.data import Cover
from janggu import Janggu
from janggu.model import input_attribution
from janggu.data import plotGenomeTrack
from janggu.data import LineTrack
from janggu.data import SeqTrack

from data_utils import get_data


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

shared_space = {
    'type': 'dnase_dna',
    'seq_dropout': True,
    'dnaflank': 150,
    'order': 3,
    'stranded': 'double',
    'nmotifs1': 10,
    'motiflen': 11,
    'pool1': 30,
    'nmotifs2': 8,
    'hypermotiflen': 3,
    'dnaseflank': 450,
    'normalize': 'tpm',
    'augment': 'orient',
    'nkernel1': 10,
    'kernel1len': 5,
    'kernel1pool': 2,
    'nkernel2': 5,
    'kernel2len': 3,
    'binsize': 200,
#concat': 'flatten',
    'inception': True,
    'epochs': 100,
    'opt': 'sgd'
    }

train_data, val_data, test_data = get_data(shared_space)

name = 'randominit_dnase_dna_joint_model_dnase_run_zscorelog_orient_7_1_dna_o3_d0.2_run_5_1'

model = Janggu.create_by_name(name)
model.compile(optimizer='sgd', loss='binary_crossentropy',
              metrics=['acc'])

# run evaluation with callbacks
pred = model.evaluate(test_data[0], test_data[1],
                      datatags=['chr2'], callbacks=['prc', 'auprc'])

# convert the prediction to a cover object
pred = model.predict(test_data[0])

# convert predictions to Coverage track
cov_pred = Cover.create_from_array('predict', pred,
                                   test_data[1].gindexer,
                                   conditions=['JunD'],
                                   store_whole_genome=True)

chrom = 'chr3'
start = 2353000
end = 2357000
infl = input_attribution(model, test_data[0], chrom, start, end)

# plot only input and output
fig = plotGenomeTrack([LineTrack(cov_pred, color='c', marker=','),
                 LineTrack(test_data[1].data, color='b', marker=','),
                 LineTrack(test_data[0][1], color='r')],
                chrom, start, end,
                figsize=(8, 4))

fig.savefig(os.path.join(os.environ['JANGGU_OUTPUT'], 'jund_input_outout_line.png'))
fig.savefig(os.path.join(os.environ['JANGGU_OUTPUT'], 'jund_input_outout_line.eps'))
fig.savefig(os.path.join(os.environ['JANGGU_OUTPUT'], 'jund_input_outout_line.svg'))


# Plot integrated gradients for DNA sequence
chrom = 'chr3'
start = 2354950
end = 2355050

fig = plotGenomeTrack(infl[0], chrom, start, end,
                plottypes=['seqplot'],
                figsize=(6, 2))

fig.savefig(os.path.join(os.environ['JANGGU_OUTPUT'], 'jund_input_attribution_dna.png'))
fig.savefig(os.path.join(os.environ['JANGGU_OUTPUT'], 'jund_input_attribution_dna.eps'))
fig.savefig(os.path.join(os.environ['JANGGU_OUTPUT'], 'jund_input_attribution_dna.svg'))
