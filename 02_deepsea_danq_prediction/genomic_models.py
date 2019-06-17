from keras.layers import Dropout, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Bidirectional
from keras.layers import Reshape

from janggu import DnaConv2D
from janggu import inputlayer
from janggu import outputdense

@inputlayer
@outputdense('sigmoid')
def deepsea_model(inputs, inp, oup, params):
    with inputs.use('dna') as dna_in:
        layer = dna_in
    cl = Conv2D(320, (8, 1), activation='relu', name='dna_conv2d_1')
    if params['stranded'] == 'double':
        layer = DnaConv2D(cl, name='dna_dnaconv2d_2')(layer)
    else:
        layer = cl(layer)

    layer = MaxPooling2D((4, 1), name='dna_maxpooling1')(layer)
    layer = Dropout(0.2)(layer)

    layer = Conv2D(480, (8, 1), name='dna_conv2d_2', activation='relu')(layer)
    layer = MaxPooling2D((4, 1), name='dna_maxpooling2')(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv2D(960, (8, 1), name='dna_conv2d_3', activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Flatten()(layer)
    layer = Dense(925, activation='relu')(layer)
    return inputs, layer


@inputlayer
@outputdense('sigmoid')
def danq_model(inputs, inp, oup, params):
    with inputs.use('dna') as dna_in:
        layer = dna_in
    cl = Conv2D(320, (26, 1), activation='relu', name='dna_conv2d_1')
    if params['stranded'] == 'double':
        layer = DnaConv2D(cl, name='dna_dnaconv2d_2')(layer)
    else:
        layer = cl(layer)

    layer = MaxPooling2D((13, 1), name='dna_maxpooling1')(layer)

    layer = Dropout(0.2)(layer)

    newdim = tuple(layer.shape.as_list()[1:2]) + tuple(layer.shape.as_list()[-1:])
    layer = Reshape(newdim)(layer)

    layer = Bidirectional(LSTM(320, return_sequences=True))(layer)

    layer = Dropout(0.5)(layer)

    layer = Flatten()(layer)

    layer = Dense(925, activation='relu')(layer)
    return inputs, layer

