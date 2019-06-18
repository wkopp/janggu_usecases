import argparse
import os
import numpy as np
import pandas as pd

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Concatenate
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import average_precision_score

from janggu import Janggu
from janggu import model_from_json
from data_utils import get_data
from model_utils import get_opt


np.random.seed(1234)

# Fetch parser arguments
PARSER = argparse.ArgumentParser(description='Command description.')

PARSER.add_argument('-inputpath', dest='inpath',
                    default='../data', help='Location of input files')

PARSER.add_argument('-path', dest='path',
                    default='../jund_results',
                    help="Output directory for the examples.")

dnasemodelname = 'dnase_run_zscorelog_orient_7_{}'
dnamodelname = 'dna_o3_d0.2_run_5_{}'

args = PARSER.parse_args()

os.environ['JANGGU_OUTPUT'] = args.path

inpath = args.inpath

print("#" * 20)
print("Fitting combined models ...")


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
    'concat': 'flatten',
    'inception': True,
    'epochs': 100,
    'opt': 'sgd'
    }

space = {'modeltype': shared_space}

auprc_pre_val = []
auprc_pre_test = []
auprc_rand_val = []
auprc_rand_test = []

# Next, we concatenate the individual models and fine-tune them.
# Furthermore, the combined models are reset with random weights and trained from scratch
# as a comparison.
for dnarun, dnaserun in zip([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]):
    # load pre-trained models
    dnaname = dnamodelname.format(dnarun)
    dnasename = dnasemodelname.format(dnaserun)
    dnamodel = Janggu.create_by_name(dnaname)
    dnasemodel = Janggu.create_by_name(dnasename)

    # remove output layer, concatenate the top-hidden layers, append output
    hidden_dna = dnamodel.kerasmodel.layers[-2].output
    hidden_dnase = dnasemodel.kerasmodel.layers[-2].output

    joint_hidden = Concatenate(name='concat')([hidden_dna, hidden_dnase])
    output = Dense(1, activation='sigmoid', name='peaks')(joint_hidden)

    # fit the model with preinitialized weights
    jointmodel = Janggu(dnamodel.kerasmodel.inputs +  dnasemodel.kerasmodel.inputs,
                        output,
                        name='pretrained_dnase_dna_joint_model_{}_{}'.format(dnasename, dnaname))

    # reload the same model architecture, but this will
    # randomly reinitialized the weights
    newjointmodel = model_from_json(jointmodel.kerasmodel.to_json())

    newjointmodel = Janggu(newjointmodel.inputs,
                           newjointmodel.outputs,
                           name='randominit_dnase_dna_joint_model_{}_{}'.format(dnasename, dnaname))
    newjointmodel.compile(optimizer=get_opt('amsgrad'), loss='binary_crossentropy',
                          metrics=['acc'])

    hist = newjointmodel.fit(train_data[0], train_data[1],
                             epochs=shared_space['epochs'], batch_size=64,
                             validation_data=val_data,
                             callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    pred_test = newjointmodel.predict(test_data[0])
    pred_val = newjointmodel.predict(val_data[0])

    auprc_val = average_precision_score(val_data[1][:], pred_val)
    auprc_test = average_precision_score(test_data[1][:], pred_test)
    print('auprc_val: {:.2%}'.format(auprc_val))
    print('auprc_test: {:.2%}'.format(auprc_test))
    auprc_rand_val.append(auprc_val)
    auprc_rand_test.append(auprc_test)


df = pd.DataFrame({'auprc_val': auprc_rand_val, 'auprc_test': auprc_rand_test})

df.to_csv(os.path.join(os.environ['JANGGU_OUTPUT'],
                       "dnase_dna_use_randominit_submodels.tsv"), sep='\t')
