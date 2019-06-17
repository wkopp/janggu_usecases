import sys
import traceback
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalMaxPooling2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Concatenate
from keras import backend as K

from keras.callbacks import EarlyStopping
from sklearn.metrics import average_precision_score
from keras import optimizers
from janggu.layers import DnaConv2D
from janggu import inputlayer
from janggu import Janggu
from data_utils import get_data

def get_opt(name):
    if name == 'amsgrad':
        opt = optimizers.Adam(amsgrad=True, clipvalue=.5, clipnorm=1.)
    elif name == 'sgd':
        opt = optimizers.SGD(lr=0.01, decay=1e-6,
                             momentum=0.9, nesterov=True,
                             clipvalue=.5, clipnorm=1.)
    else:
        opt = optimizers.RMSprop(clipvalue=.5, clipnorm=1.)
    return opt

@inputlayer
def dna_model(inputs, inp, oup, params):
    with inputs.use('dna') as dna_in:
        layer = dna_in

    if params['seq_dropout'] > 0.0:
        layer = Dropout(params['seq_dropout'])(layer)

    cl = Conv2D(params['nmotifs1'], (params['motiflen'], 1), activation='relu')
    if params['stranded'] == 'double':
        layer = DnaConv2D(cl)(layer)
    else:
        layer = cl(layer)

    layer = MaxPooling2D((params['pool1'], 1))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(params['nmotifs2'], (params['hypermotiflen'], 1), activation='relu')(layer)
    return inputs, layer

@inputlayer
def dnase_model(inputs, inp, oup, params):
    with inputs.use('dnase') as dnase_in:
        layer = dnase_in

    cl = Conv2D(params['nkernel1'], (params['kernel1len'], 2),
                activation='relu', name='dnase_conv_1')
    layer = cl(layer)

    layer = MaxPooling2D((params['kernel1pool'], 1), name='dnase_maxpooling_1')(layer)
    layer = Conv2D(params['nkernel2'],
                   (params['kernel2len'], 1), activation='relu', name='dnase_conv_2')(layer)
    return inputs, layer

@inputlayer
def dnase_dna_model(inputs, inp, oup, params):
    xin_dna, dna_hidden = dna_model(inputs, inp, oup, params)
    xin_dnase, dnase_hidden = dnase_model(inputs, inp, oup, params)
    if params['concat'] == 'flatten':
        dna_hidden = Flatten()(dna_hidden)
        dnase_hidden = Flatten()(dnase_hidden)
    else:
        dna_hidden = GlobalMaxPooling2D()(dna_hidden)
        dnase_hidden = GlobalMaxPooling2D()(dnase_hidden)

    layer_c = Concatenate()([dna_hidden, dnase_hidden])

    layer_1 = Dense(32, activation='relu')(layer_c)
    layer_2 = Dense(16, activation='relu')(layer_1)

    if params['inception'] == True:
        layer = Concatenate()([layer_c, layer_1, layer_2])
    else:
        layer = layer_2

    return inputs, layer

@inputlayer
def get_model(inputs, inp, oup, params):
    if params['type'] == 'dna_only':
        xin, layer = dna_model(inputs, inp, oup, params)
        layer = GlobalMaxPooling2D(name='dna_global_max')(layer)
        layer = BatchNormalization(name='dna_batch_2')(layer)
    elif params['type'] == 'dnase_dna':
        xin, layer = dnase_dna_model(inputs, inp, oup, params)
    elif params['type'] in ['dnase_only', 'dnase_bam_only']:
        xin, layer = dnase_model(inputs, inp, oup, params)
        layer = GlobalMaxPooling2D(name='dnase_global_max')(layer)
        layer = BatchNormalization(name='dnase_batch_2')(layer)
    output = Dense(1, activation='sigmoid', name='peaks')(layer)
    return inputs, output


def objective(params):
    train_data, val_data, test_data = get_data(params)
    # define a keras model only based on DNA

    try:
        K.clear_session()
        model = Janggu.create(get_model, params, train_data[0], train_data[1], name=params['name'])
        model.compile(optimizer=get_opt(params['opt']), loss='binary_crossentropy',
                      metrics=['acc'])
        hist = model.fit(train_data[0], train_data[1], epochs=params['epochs'], batch_size=64,
                         validation_data=val_data,
                         callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    except ValueError:
        traceback.print_stack()
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(repr(traceback.extract_tb(exc_traceback)))
        return {'status': 'fail'}
    print('#' * 40)
    for key in hist.history:
        print('{}: {}'.format(key, hist.history[key][-1]))
    print('#' * 40)
    pred_test = model.predict(test_data[0])
    pred_val = model.predict(val_data[0])

    model.evaluate(val_data[0], val_data[1], callbacks=['auprc', 'auroc'], datatags=['val'])
    model.evaluate(test_data[0], test_data[1], callbacks=['auprc', 'auroc'], datatags=['test'])

    auprc_val = average_precision_score(val_data[1][:], pred_val)
    auprc_test = average_precision_score(test_data[1][:], pred_test)
    model.summary()
    print('auprc_val: {:.2%}'.format(auprc_val))
    print('auprc_test: {:.2%}'.format(auprc_test))
    return {'loss': hist.history['val_loss'][-1], 'status': 'ok', 'all_losses': hist.history,
            'auprc_val': auprc_val,
            'auprc_test': auprc_test,
            'model_config': model.kerasmodel.to_json(),
            'model_weights': model.kerasmodel.get_weights(),
            'concrete_params': params,
            'modelname': model.name}

