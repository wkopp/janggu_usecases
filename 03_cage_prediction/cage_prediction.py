import argparse
import os
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from itertools import product

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from keras import optimizers
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalMaxPooling2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Concatenate

from janggu.data import Bioseq
from janggu.data import Cover
from janggu.data import ReduceDim
from janggu.data import split_train_test
from janggu.data.genomicarray import LogTransform
from janggu.data.genomicarray import ZScore

from janggu import Janggu
from janggu import inputlayer, outputdense
from janggu import model_from_json
from janggu.layers import DnaConv2D


PARSER = argparse.ArgumentParser(description='DNA model.')

PARSER.add_argument('-order', dest='order',
                    default=1, type=int, help='One-hot encoding order')
PARSER.add_argument('-inputpath', dest='inputpath',
                    default='../data', type=str, help='Input path')
PARSER.add_argument('-outputpath', dest='outputpath',
                    default='../', type=str, help='Output path')

args = PARSER.parse_args()


dnaorder = args.order
np.random.seed(1234)

os.environ['JANGGU_OUTPUT'] = os.path.join(args.outputpath,
                                           'results_cage_promoters_order{}'.format(dnaorder))

inpath = args.inputpath

ROI_INPUT_TRAIN = os.path.join(inpath, 'gencode.v29.tss.gtf')

# ref genome
REFGENOME = os.path.join(inpath, 'hg38.fa')

# input training
DNASE = os.path.join(inpath, 'dnase.{}.bam')
H3K4me3 = os.path.join(inpath, 'h3k4me3.{}.bigWig')

# output training
RNA = os.path.join(inpath, 'cage.{}.{}.bam')

def get_opt(name):
    if name == 'amsgrad':
        opt = optimizers.Adam(amsgrad=True, clipvalue=.5, clipnorm=1.)
    elif name == 'sgd':
        opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9,
                             nesterov=True, clipvalue=.5, clipnorm=1.)
    else:
        opt = optimizers.RMSprop(clipvalue=.5, clipnorm=1.)
    return opt


def get_data(params):
    zscore = ZScore()
    LABELS_TRAIN = ReduceDim(Cover.create_from_bam('geneexpr',
                                                   bamfiles=RNA.format(
                                                       params['traincell'], params['trainrep']),
                                                   roi=ROI_INPUT_TRAIN,
                                                   flank=params['cageflank'],
                                                   conditions=['GeneExpr'],
                                                   resolution=None,
                                                   store_whole_genome=False,
                                                   storage='ndarray',
                                                   normalizer=[LogTransform(), zscore],
                                                   stranded=False,
                                                   cache=True), aggregator="mean")
    train_labels = LABELS_TRAIN
    train_input = []
    if params['inputs'] in ['dna_only', 'epi_dna']:
        dnaflank = params['dnaflank']
        order = params['order']
        # DNA
        DNA_TRAIN = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                                 roi=ROI_INPUT_TRAIN,
                                                 flank=dnaflank,
                                                 order=order,
                                                 cache=True,
                                                 store_whole_genome=False)
        train_input += [DNA_TRAIN]
    if params['inputs'] in ['epi_only', 'epi_dna']:
        zscore = ZScore()
        dnase_TRAIN = ReduceDim(Cover.create_from_bam('dnase',
                                                      bamfiles=DNASE.format(params['traincell']),
                                                      roi=ROI_INPUT_TRAIN,
                                                      flank=params['dnaseflank'],
                                                      resolution=None,
                                                      store_whole_genome=False,
                                                      normalizer=[LogTransform(), zscore],
                                                      cache=True), aggregator="mean")
        train_input += [dnase_TRAIN]
        zscore = ZScore()
        h3k4_TRAIN = ReduceDim(Cover.create_from_bigwig('h3k4',
                                                        bigwigfiles=[
                                                            H3K4me3.format(params['traincell'])],
                                                        roi=ROI_INPUT_TRAIN,
                                                        flank=params['dnaseflank'],
                                                        store_whole_genome=False,
                                                        normalizer=[LogTransform(), zscore],
                                                        cache=True), aggregator="mean")
        train_input += [h3k4_TRAIN]
    if len(train_input) == 0:
        raise ValueError('no input')
    return (train_input, train_labels)


# load the dataset

@inputlayer
def dna_model_(inputs, inp, oup, params):
    with inputs.use('dna') as dna_in:
        layer = dna_in
    layer = Dropout(params['seq_dropout'], name='dna_dropout_1')(layer)
    cl = Conv2D(params['nmotifs1'], (params['motiflen'], 1),
                activation='relu', name='dna_conv2d_1')
    if params['stranded'] == 'double':
        layer = DnaConv2D(cl, name='dna_dnaconv2d_2')(layer)
    else:
        layer = cl(layer)
    layer = MaxPooling2D((params['pool1'], 1), name='dna_maxpooling1')(layer)
    layer = BatchNormalization(name='dna_batchnorm_1')(layer)
    layer = Conv2D(params['nmotifs2'], (params['hypermotiflen'], 1),
                   activation='relu',
                   name='dna_conv2d_2')(layer)
    layer = GlobalMaxPooling2D(name='global_max_pooling')(layer)
    layer = BatchNormalization(name='dna_batchnorm_2')(layer)
    return inputs, layer

@inputlayer
def epi_model_(inputs, inp, oup, params):
    layer = []
    with inputs.use('dnase') as dnase_in:
        layer += [dnase_in]
    with inputs.use('h3k4') as h3k4_in:
        layer += [h3k4_in]

    layer = Concatenate(name='epi_concat')(layer)
    layer = BatchNormalization(name='epi_batchnorm')(layer)
    return inputs, layer

@inputlayer
def joint_model_(inputs, inp, oup, params):
    _, epi_hidden = epi_model_(inputs, inp, oup, params)
    _, dna_hidden = dna_model_(inputs, inp, oup, params)
    layer = Concatenate()([epi_hidden, dna_hidden])
    return inputs, layer

@inputlayer
@outputdense('linear')
def get_model(inputs, inp, oup, params):
    if params['inputs'] == 'dna_only':
        _, layer = dna_model_(inputs, inp, oup, params)
    elif params['inputs'] == 'epi_only':
        _, layer = epi_model_(inputs, inp, oup, params)
    else:
        _, layer = joint_model_(inputs, inp, oup, params)
    return inputs, layer

val_chroms = ['chr{}'.format(i) for i in range(2, 23)]
test_chrom = 'chr1'

main_logger = logging.getLogger('rna_predict')
def objective(params):
    print(params)
    try:
        train_data = get_data(params)
        train_data, test = split_train_test(train_data, [test_chrom])
        train, val = split_train_test(train_data, [params['val_chrom']])
        # define a keras model only based on DNA
        K.clear_session()
        if params['inputs'] == 'epi_dna':
            dnam = Janggu.create_by_name('cage_promoters_dna_only')
            epim = Janggu.create_by_name('cage_promoters_epi_only')
            layer = Concatenate()([dnam.kerasmodel.layers[-2].output,
                                   epim.kerasmodel.layers[-2].output])
            layer = Dense(1, name='geneexpr')(layer)
            model = Janggu([dnam.kerasmodel.input] + epim.kerasmodel.input,
                           layer, name='cage_promoters_epi_dna')

            if not params['pretrained']:
                # This part randomly reinitializes the network
                # so that we can train it from scratch
                newjointmodel = model_from_json(model.kerasmodel.to_json())

                newjointmodel = Janggu(newjointmodel.inputs,
                                       newjointmodel.outputs,
                                       name='cage_promoters_epi_dna_randominit')
                model = newjointmodel
        else:
            model = Janggu.create(get_model, params, train_data[0],
                                  train_data[1],
                                  name='cage_promoters_{}'.format(params['inputs']))
    except ValueError:
        main_logger.exception('objective:')
        return {'status': 'fail'}
    model.compile(optimizer=get_opt(params['opt']), loss='mae', metrics=['mse'])
    hist = model.fit(train_data[0], train_data[1], epochs=params['epochs'], batch_size=64,
                     validation_data=[params['val_chrom']],
                     callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    print('#' * 40)
    for key in hist.history:
        print('{}: {}'.format(key, hist.history[key][-1]))
    print('#' * 40)
    pred_train = model.predict(train[0])
    pred_val = model.predict(val[0])
    pred_test = model.predict(test[0])
    model.evaluate(train[0], train[1],
                   callbacks=['var_explained', 'mse', 'mae', 'cor'],
                   datatags=['train'])
    mae_val = model.evaluate(val[0], val[1],
                             callbacks=['var_explained', 'mse', 'mae', 'cor'],
                             datatags=['val'])
    mae_val = mae_val[0]
    model.evaluate(test[0], test[1],
                   callbacks=['var_explained', 'mse', 'mae', 'cor'],
                   datatags=['test'])

    cor_train = np.corrcoef(train[1][:][:, 0], pred_train[:, 0])[0, 1]
    cor_val = np.corrcoef(val[1][:][:, 0], pred_val[:, 0])[0, 1]
    cor_test = np.corrcoef(test[1][:][:, 0], pred_test[:, 0])[0, 1]

    model.summary()
    main_logger.info('cor [train/val/test]: {:.2f}/{:.2f}/{:.2f}'.format(
        cor_train, cor_val, cor_test))
    return {'loss': mae_val, 'status': 'ok', 'all_losses': hist.history,
            'cor_train': cor_train,
            'cor_val': cor_val,
            'cor_test': cor_test,
            'model_config': model.kerasmodel.to_json(),
            'model_weights': model.kerasmodel.get_weights(),
            'concrete_params': params}

# first do an exhaustive grid search

print("#" * 20)
print("Test effect of scanning single or both strands and higher-order motifs")

shared_space = {
    'seq_dropout': 0.2,
    'dnaflank': 350,
    'nmotifs1': 10,
    'motiflen': 15,
    'pool1': 5,
    'nmotifs2': 8,
    'hypermotiflen': 5,
    'dnaseflank': 200,
    'inception': False,
    'traincell': 'hepg2',
    'trainrep': 'rep1',
    'cageflank': 400,
    'opt': 'amsgrad',
    'epochs': 100,
}

results = {'run':[], 'val_chrom':[], 'inputs':[], 'dnaorder':[], 'strand':[],
           'cor':[],
           'cor_val':[],
           'pretrained': []}

orders = [dnaorder]
strands = ['double']

def write_results(params, res):
    results['run'].append(params['run'])
    results['val_chrom'].append(params['val_chrom'])
    results['dnaorder'].append(params['order'])
    results['inputs'].append(params['inputs'])
    results['strand'].append(params['stranded'])
    results['cor'].append(res['cor_test'])
    results['cor_val'].append(res['cor_val'])
    results['pretrained'].append('pretrained' if params['pretrained'] else 'randominit')
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.environ['JANGGU_OUTPUT'], "gridsearch_cage_prediction.tsv"), sep='\t')

run = 0
for val_chrom, order, strand in product(val_chroms, orders, strands):
    run += 1
    shared_space['run'] = run
    shared_space['val_chrom'] = val_chrom
    shared_space['order'] = order
    shared_space['pretrained'] = False
    shared_space['seq_dropout'] = 0.2
    if order == 1:
        shared_space['seq_dropout'] = 0.0
    shared_space['stranded'] = strand

    shared_space['inputs'] = 'dna_only'
    res = objective(shared_space)
    write_results(shared_space, res)

    shared_space['inputs'] = 'epi_only'
    res = objective(shared_space)
    write_results(shared_space, res)

    shared_space['inputs'] = 'epi_dna'
    shared_space['pretrained'] = True
    res = objective(shared_space)
    write_results(shared_space, res)

    shared_space['inputs'] = 'epi_dna'
    shared_space['pretrained'] = False
    res = objective(shared_space)
    write_results(shared_space, res)


#shared_space['run'] = run
shared_space['val_chrom'] = "chr22"
shared_space['order'] = dnaorder
shared_space['pretrained'] = False
shared_space['seq_dropout'] = 0.2
shared_space['inputs'] = 'epi_dna'
params = shared_space
train_data = get_data(params)
train, test = split_train_test(train_data, [test_chrom])

model = Janggu.create_by_name('cage_promoters_epi_dna')

testpred = model.predict(test[0])


fig, ax = plt.subplots()
ax.scatter(test[1][:], testpred)
ax.set_xlabel('Observed normalized CAGE signal')
ax.set_ylabel('Predicted normalized CAGE signal')
fig.savefig(os.path.join(os.environ['JANGGU_OUTPUT'], 'cage_promoter_testchrom_agreement.png'))
