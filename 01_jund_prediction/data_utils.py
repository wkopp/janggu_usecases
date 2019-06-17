import os

from janggu.data import Bioseq
from janggu.data import Cover
from janggu.data import ReduceDim
from janggu.data import RandomOrientation
from janggu.data import RandomSignalScale
from janggu.data import split_train_test

inpath = '../data'

# ref genome
REFGENOME = os.path.join(inpath, 'hg38.fa')

# training and test roi
ROI = os.path.join(inpath, 'trim_roi_jund_extended.bed')

# jund peaks (labels)
PEAKS = os.path.join(inpath, 'jund_raw_peaks.bed')

# dnase-seq fold-enrichment
DNASE_STAM_ROADMAP = os.path.join(inpath, 'dnase_stam_roadmap.bam')
DNASE_STAM_ENCODE = os.path.join(inpath, 'dnase_stam_encode.bam')

def get_data(params):
    binsize = params['binsize']

    # PEAKS
    LABELS = ReduceDim(Cover.create_from_bed('peaks',
                                             bedfiles=PEAKS,
                                             roi=ROI,
                                             binsize=binsize,
                                             conditions=['JunD'],
                                             resolution=binsize,
                                             store_whole_genome=True,
                                             storage='sparse',
                                             cache=True), aggregator='max')

    # training on chr1, validation on chr2, test on chr3 with swapped Dnase samples
    LABELS, LABELS_TEST = split_train_test(LABELS, 'chr3')
    LABELS_TRAIN, LABELS_VAL = split_train_test(LABELS, 'chr2')
    if params['type'] in ['dna_only', 'dnase_dna']:
        dnaflank = params['dnaflank']
        order = params['order']
        # DNA
        DNA = Bioseq.create_from_refgenome('dna', refgenome=REFGENOME,
                                           roi=ROI,
                                           binsize=binsize,
                                           flank=dnaflank,
                                           order=order,
                                           cache=True,
                                           store_whole_genome=True)

        DNA, DNA_TEST = split_train_test(DNA, 'chr3')
        DNA_TRAIN, DNA_VAL = split_train_test(DNA, 'chr2')
    if params['type'] in ['dnase_bam_only', 'dnase_dna']:

        dnaseflank = params['dnaseflank']
        # ACCESSIBILITY
        ACCESS_TEST = Cover.create_from_bam('dnase',
                                            bamfiles=[DNASE_STAM_ENCODE, DNASE_STAM_ROADMAP],
                                            roi=ROI,
                                            binsize=binsize,
                                            conditions=['Encode', 'Roadmap'],
                                            flank=dnaseflank,
                                            resolution=50,
                                            normalizer=params['normalize'],
                                            store_whole_genome=True,
                                            cache=True)
        ACCESS = Cover.create_from_bam('dnase', roi=ROI,
                                       bamfiles=[DNASE_STAM_ROADMAP, DNASE_STAM_ENCODE],
                                       binsize=binsize,
                                       conditions=['Roadmap', 'Encode'],
                                       resolution=50,
                                       flank=dnaseflank,
                                       normalizer=params['normalize'],
                                       store_whole_genome=True,
                                       cache=True)

        _, ACCESS_TEST = split_train_test(ACCESS_TEST, 'chr3')
        ACCESS, _ = split_train_test(ACCESS, 'chr3')
        ACCESS_TRAIN, ACCESS_VAL = split_train_test(ACCESS, 'chr2')

    if params['type'] in ['dna_dnase', 'dnase_bam_only']:
        if params['augment'] == 'orient':
            ACCESS_TRAIN = RandomOrientation(ACCESS_TRAIN)
        if params['augment'] == 'scale':
            ACCESS_TRAIN = RandomSignalScale(ACCESS_TRAIN, 0.1)
        if params['augment'] == 'both':
            ACCESS_TRAIN = RandomSignalScale(RandomOrientation(ACCESS_TRAIN), 0.1)

    if params['type'] == 'dna_only':
        return (DNA_TRAIN, LABELS_TRAIN), (DNA_VAL, LABELS_VAL), \
               (DNA_TEST, LABELS_TEST)
    elif params['type'] == 'dnase_dna':
        return ([DNA_TRAIN, ACCESS_TRAIN], LABELS_TRAIN), \
                ([DNA_VAL, ACCESS_VAL], LABELS_VAL),\
               ([DNA_TEST, ACCESS_TEST], LABELS_TEST)
    elif params['type'] in ['dnase_bam_only']:
        return ([ACCESS_TRAIN], LABELS_TRAIN), \
               ([ACCESS_VAL], LABELS_VAL), \
               ([ACCESS_TEST], LABELS_TEST)


