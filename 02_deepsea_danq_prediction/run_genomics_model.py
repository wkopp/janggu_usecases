import argparse
import os

from janggu.data import Cover
from janggu.data import Bioseq
from janggu.data import GenomicIndexer
from janggu.data import ReduceDim
from janggu.data import view
from janggu import Janggu

from genomic_models import deepsea_model, danq_model
from keras import optimizers
from keras.callbacks import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument('-data', dest='data', type=str, default='./data', help='Raw data folder')
parser.add_argument('-o', dest='order', type=int, default=3, help='Sequence features order')
parser.add_argument('-epochs', dest='epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('-rep', dest='rep', type=int, default=1, help='Repetition of the experiment.')
parser.add_argument('-model', dest='model', type=str, default='danq',
                    help='Model type: dsea or danq.')
parser.add_argument('-strand', dest='strand', type=str, default='double',
                    help='Strandedness to consider')
parser.add_argument('-dev', dest='device', type=str, default="0", help='Device ID of gpu.')
parser.add_argument('-flank', dest='flank', type=int, default=900, help='Flank')
parser.add_argument('-evaluate', dest='evaluate', default=False, action='store_true',
                    help='Reevaluate the previously trained models')

args = parser.parse_args()

datadir = args.data
order = args.order
epochs = args.epochs
rep = args.rep
device = args.device
evaluate = args.evaluate
flank = args.flank
modelname = args.model
strand = args.strand

os.environ["CUDA_VISIBLE_DEVICES"] = device

bedfilelist = os.path.join('..', 'extra', 'narrowpeakfiles.txt')

with open(bedfilelist, 'r') as f:
    lines = f.readlines()

bedfiles = [os.path.join(datadir, b.strip()) for b in lines]

refgenome = os.path.join(datadir, 'hg19.fa')

train_roi = os.path.join('..', 'extra', 'deepsea_train_train.bed')
val_roi = os.path.join('..', 'extra', 'deepsea_train_valid.bed')
test_roi = os.path.join('..', 'extra', 'test.bed')


os.environ['JANGGU_OUTPUT'] = './deepsea_results'

def get_data(params):
    train_labels = Cover.create_from_bed('labels', bedfiles=bedfiles, roi=train_roi,
                                                   resolution=200,
                                                   store_whole_genome=True,
                                                   storage='sparse', cache=True,
                                                   dtype='int8',
                                                   minoverlap=.5, verbose=True)
    test_labels = view(train_labels, test_roi)
    val_labels = view(train_labels, val_roi)
    train_seq = Bioseq.create_from_refgenome('dna', refgenome=refgenome, roi=train_roi,
                                             store_whole_genome=True,
                                             storage='ndarray', cache=True,
                                             order=params['order'],
                                             flank=params['flank'], verbose=True)
    test_seq = view(train_seq, test_roi)
    val_seq = view(train_seq, val_roi)
    return ((train_seq, ReduceDim(train_labels)), (val_seq, ReduceDim(val_labels)), (test_seq, ReduceDim(test_labels)))

version = 'r{}'.format(rep)

if modelname == 'dsea':
    dna_model = deepsea_model
elif modelname == 'danq':
    dna_model = danq_model

def get_opt(name):
    if name == 'amsgrad':
        opt = optimizers.Adam(amsgrad=True, clipvalue=.5, clipnorm=1.)
    elif name == 'sgd':
        opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9,
                             nesterov=True, clipvalue=.5, clipnorm=1.)
    else:
        opt = optimizers.RMSprop(clipvalue=.5, clipnorm=1.)
    return opt

flatten = True

pars = {'order': order,
        'stranded': strand,
        'flank': flank,
        'rep': 'r{}'.format(rep),
        'flatten': flatten}


DATA = get_data(pars)
mname = '{}_s{}_o{}_f{}_a{}_r{}'.format(modelname, pars['stranded'], pars['order'],
                                        pars['flank'], pars['flatten'], pars['rep'])

if not evaluate:
    model = Janggu.create(dna_model, pars,
                          inputs=DATA[0][0], outputs=DATA[0][1],
                          name=mname)
    model.summary()

    model.compile(optimizer=get_opt('amsgrad'), loss='binary_crossentropy', metrics=['accuracy'])

    train_data = DATA[0]
    val_data = DATA[1]
    test_data = DATA[2]
    hist = model.fit(train_data[0], train_data[1],
                     epochs=epochs, batch_size=128,
                     validation_data=val_data,
                     callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    model.evaluate(test_data[0], test_data[1], callbacks=['auc', 'auprc'])
if evaluate:
    test_data = DATA[2]
    model = Janggu.create_by_name(mname)
    model.compile(optimizer=get_opt('amsgrad'), loss='binary_crossentropy', metrics=['accuracy'])

    model.evaluate(test_data[0], test_data[1], callbacks=['auc', 'auprc'])
