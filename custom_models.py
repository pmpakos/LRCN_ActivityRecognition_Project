# from keras import layers
from keras import models
from keras import layers
from keras import initializers
from keras import optimizers
from keras import regularizers
from keras import constraints

from LRN2D import *

weight_decay = 0.0005

def CaffeDonahue(input_shape, num_labels):
    model = models.Sequential()

    #### WEIGHT FILLER - KERNEL INITIALIZER ?????
    # weight filler = gaussian std=0.01
    # kernel initializer = default 'glorot_uniform'
    # https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead
    ####

    #Conv1, Relu1
    model.add(layers.Conv2D(filters=96,
                            kernel_size=(7,7),
                            strides=(2,2),
                            padding='valid', #maybe same 
                            activation='relu',
                            use_bias=True,
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                            bias_initializer=initializers.Constant(value=0.1),
                            kernel_regularizer=regularizers.l2(weight_decay),
                            bias_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None,
                            input_shape=input_shape,
                            name='conv1'))

    #Pool1
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='pool1'))

    #### LRN NORMALIZATION ???
    # Edw pi8anws 8a xreiastei na kanw to LocalResponseNormalization, pou 
    # twra exei afaire8ei ap to keras
    # Sto caffe eixe local_size=5, alpha=0.0001, beta=0.75
    ####
    #Norm1
    # model.add(layers.BatchNormalization(name='BatchNorm1'))
    model.add(LRN2D(alpha=0.0001, beta=0.75, n=5, name='norm1'))

    #### GROUPING
    # group (g): we restrict the connectivity of each filter to a subset
    # of the input. Specifically, the input and output channels are
    # separated into g groups, and the ith output group channels will
    # be only connected to the i-th input group channels.
    # group can reduce resource use, while helping to preserve accuracy.
    #
    # https://stackoverflow.com/a/40876685
    #
    # 8elei group=2
    # https://groups.google.com/forum/?utm_medium=email&utm_source=footer#!topic/keras-users/bxPA4_Bda14
    ####
    
    #Conv2, Relu2
    model.add(layers.Conv2D(filters=384,
                            kernel_size=(5,5),
                            strides=(2,2),
                            padding='valid', #maybe same 
                            activation='relu',
                            use_bias=True,
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                            bias_initializer=initializers.Constant(value=0.1),
                            kernel_regularizer=regularizers.l2(weight_decay),
                            bias_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None,
                            name='conv2'))

    #Pool2
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name='pool2'))

    #Norm2
    model.add(LRN2D(alpha=0.0001, beta=0.75, n=5, name='norm2'))

    # Anti gia padding 1, kanw padding='same'
    # Gia to padding : https://groups.google.com/forum/?utm_medium=email&utm_source=footer#!topic/keras-users/bxPA4_Bda14

    #Conv3, Relu3
    model.add(layers.Conv2D(filters=512,
                            kernel_size=(3,3),
                            padding='same',
                            activation='relu',
                            use_bias=True,
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                            bias_initializer=initializers.Constant(value=0.1),
                            kernel_regularizer=regularizers.l2(weight_decay),
                            bias_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None,
                            name='conv3'))
    #Conv4, Relu4
    model.add(layers.Conv2D(filters=512,
                            kernel_size=(3,3),
                            padding='same', 
                            activation='relu',
                            use_bias=True,
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                            bias_initializer=initializers.Constant(value=0.1),
                            kernel_regularizer=regularizers.l2(weight_decay),
                            bias_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None,
                            name='conv4'))

    #Conv5, Relu5
    model.add(layers.Conv2D(filters=384,
                            kernel_size=(3,3),
                            padding='same',
                            activation='relu',
                            use_bias=True,
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                            bias_initializer=initializers.Constant(value=0.1),
                            kernel_regularizer=regularizers.l2(weight_decay),
                            bias_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None,
                            name='conv5'))

    #Pool5
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='pool5'))

    #Flatten
    model.add(layers.Flatten(name='flatten'))

    #FC6, Relu6
    model.add(layers.Dense(units=4096,
                           activation='relu',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer=initializers.Constant(value=0.1),
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=None,
                           bias_constraint=None,
                           name='fc6'))

    #Dropout6
    model.add(layers.Dropout(rate=0.5, noise_shape=None, name='drop6'))

    #FC7, Relu7
    model.add(layers.Dense(units=4096,
                           activation='relu',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer=initializers.Constant(value=0.1),
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=None,
                           bias_constraint=None,
                           name='fc7'))

    #Dropout7
    model.add(layers.Dropout(rate=0.5, noise_shape=None, name='drop7'))

    #FC8
    model.add(layers.Dense(units=num_labels,
                           activation='softmax',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer='zeros',
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=None,
                           bias_constraint=None,
                           name='fc8-ucf'))
    
    return model


def CaffeDonahueFunctional(input_shape, num_labels):
    
    #Input
    A = layers.Input(shape=input_shape)
    #### WEIGHT FILLER - KERNEL INITIALIZER ?????
    # weight filler = gaussian std=0.01
    # kernel initializer = default 'glorot_uniform'
    # https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead
    ####

    #Conv1, Relu1
    B = layers.Conv2D(filters=96,
                      kernel_size=(7,7),
                      strides=(2,2),
                      padding='valid', #maybe same 
                      activation='relu',
                      use_bias=True,
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                      bias_initializer=initializers.Constant(value=0.1),
                      kernel_regularizer=regularizers.l2(weight_decay),
                      bias_regularizer=None,
                      kernel_constraint=constraints.max_norm(2.),
                      bias_constraint=None,
                      name='conv1')(A)

    #Pool1
    B = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='pool1')(B)

    #### LRN NORMALIZATION ???
    # Edw pi8anws 8a xreiastei na kanw to LocalResponseNormalization, pou 
    # twra exei afaire8ei ap to keras
    # Sto caffe eixe local_size=5, alpha=0.0001, beta=0.75
    ####
    #Norm1
    B = LRN2D(alpha=0.0001, beta=0.75, n=5, name='norm1')(B)

    #Conv2, Relu2 (grouped)
    #### GROUPING
    # group (g): we restrict the connectivity of each filter to a subset
    # of the input. Specifically, the input and output channels are
    # separated into g groups, and the ith output group channels will
    # be only connected to the i-th input group channels.
    # group can reduce resource use, while helping to preserve accuracy.
    #
    # https://stackoverflow.com/a/40876685
    #
    # 8elei group=2
    # https://groups.google.com/forum/?utm_medium=email&utm_source=footer#!topic/keras-users/bxPA4_Bda14
    # https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
    ####

    grouping = 2
    prev_filters = 96
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[: ,: ,: , i*step : (i+1)*step])(B)
        name = 'conv2_'+str(i)
        groups.append(layers.Conv2D(filters=384//grouping,
                                    kernel_size=(5,5),
                                    strides=(2,2),
                                    padding='valid', #maybe same 
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    bias_initializer=initializers.Constant(value=0.1),
                                    kernel_regularizer=regularizers.l2(weight_decay),
                                    bias_regularizer=None,
                                    kernel_constraint=constraints.max_norm(2.),
                                    bias_constraint=None,
                                    name=name)(group))
    C = layers.Concatenate(name='concat2')(groups)

    #Pool2
    C = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name='pool2')(C)

    #Norm2
    C = LRN2D(alpha=0.0001, beta=0.75, n=5, name='norm2')(C)

    #Conv3, Relu3
    C = layers.Conv2D(filters=512,
                      kernel_size=(3,3),
                      padding='same',
                      activation='relu',
                      use_bias=True,
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                      bias_initializer=initializers.Constant(value=0.1),
                      kernel_regularizer=regularizers.l2(weight_decay),
                      bias_regularizer=None,
                      kernel_constraint=constraints.max_norm(2.),
                      bias_constraint=None,
                      name='conv3')(C)

    #Conv4, Relu4
    grouping = 2
    prev_filters = 512
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[: ,: ,: , i*step : (i+1)*step])(C)
        name = 'conv4_'+str(i)
        groups.append(layers.Conv2D(filters=512//grouping,
                                    kernel_size=(3,3),
                                    padding='same', #maybe same 
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    bias_initializer=initializers.Constant(value=0.1),
                                    kernel_regularizer=regularizers.l2(weight_decay),
                                    bias_regularizer=None,
                                    kernel_constraint=constraints.max_norm(2.),
                                    bias_constraint=None,
                                    name=name)(group))
    D = layers.Concatenate(name='concat4')(groups)

    #Conv5, Relu5
    grouping = 2
    prev_filters = 512
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[: ,: ,: , i*step : (i+1)*step])(D)
        name = 'conv5_'+str(i)
        groups.append(layers.Conv2D(filters=384//grouping,
                                    kernel_size=(3,3),
                                    padding='same', #maybe same 
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    bias_initializer=initializers.Constant(value=0.1),
                                    kernel_regularizer=regularizers.l2(weight_decay),
                                    bias_regularizer=None,
                                    kernel_constraint=constraints.max_norm(2.),
                                    bias_constraint=None,
                                    name=name)(group))
    E = layers.Concatenate(name='concat5')(groups)

    #Pool5
    E = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='pool5')(E)

    #Flatten
    E = layers.Flatten(name='flatten')(E)


    #FC6, Relu6
    E = layers.Dense(units=4096,
                           activation='relu',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer=initializers.Constant(value=0.1),
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=constraints.max_norm(2.),
                           bias_constraint=None,
                           name='fc6')(E)

    #Dropout6
    E = layers.Dropout(rate=0.5, noise_shape=None, name='drop6')(E)

    #FC7, Relu7
    E = layers.Dense(units=4096,
                           activation='relu',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer=initializers.Constant(value=0.1),
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=constraints.max_norm(2.),
                           bias_constraint=None,
                           name='fc7')(E)

    #Dropout7
    E = layers.Dropout(rate=0.5, noise_shape=None, name='drop7')(E)

    #FC8
    E = layers.Dense(units=num_labels,
                           activation='softmax',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer='zeros',
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=constraints.max_norm(2.),
                           bias_constraint=None,
                           name='fc8-ucf')(E)

    model = models.Model(A,E)
    return model


def CaffenetDefault(input_shape, num_labels):
    #Input
    A = layers.Input(shape=input_shape)
    #### WEIGHT FILLER - KERNEL INITIALIZER ?????
    # weight filler = gaussian std=0.01
    # kernel initializer = default 'glorot_uniform'
    # https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead
    ####

    #Conv1, Relu1
    B = layers.Conv2D(filters=96,
                      kernel_size=(11,11),
                      strides=(4,4),
                      padding='valid', #maybe same 
                      activation='relu',
                      use_bias=True,
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                      bias_initializer=initializers.Constant(value=0),
                      kernel_regularizer=regularizers.l2(weight_decay),
                      bias_regularizer=None,
                      kernel_constraint=constraints.max_norm(2.),
                      bias_constraint=None,
                      name='conv1')(A)

    #Pool1
    B = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='pool1')(B)

    #### LRN NORMALIZATION ???
    # Edw pi8anws 8a xreiastei na kanw to LocalResponseNormalization, pou 
    # twra exei afaire8ei ap to keras
    # Sto caffe eixe local_size=5, alpha=0.0001, beta=0.75
    ####
    #Norm1
    B = LRN2D(alpha=0.0001, beta=0.75, n=5, name='norm1')(B)

    #Conv2, Relu2 (grouped)
    #### GROUPING
    # group (g): we restrict the connectivity of each filter to a subset
    # of the input. Specifically, the input and output channels are
    # separated into g groups, and the ith output group channels will
    # be only connected to the i-th input group channels.
    # group can reduce resource use, while helping to preserve accuracy.
    #
    # https://stackoverflow.com/a/40876685
    #
    # 8elei group=2
    # https://groups.google.com/forum/?utm_medium=email&utm_source=footer#!topic/keras-users/bxPA4_Bda14
    # https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
    ####

    grouping = 2
    prev_filters = 96
    step = prev_filters//grouping
    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[: ,: ,: , i*step : (i+1)*step])(B)
        name = 'conv2_'+str(i)
        groups.append(layers.Conv2D(filters=256//grouping,
                                    kernel_size=(5,5),
                                    padding='same', #maybe same 
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    bias_initializer=initializers.Constant(value=1),
                                    kernel_regularizer=regularizers.l2(weight_decay),
                                    bias_regularizer=None,
                                    kernel_constraint=constraints.max_norm(2.),
                                    bias_constraint=None,
                                    name=name)(group))
    C = layers.Concatenate(name='concat2')(groups)

    #Pool2
    C = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='pool2')(C)

    #Norm2
    C = LRN2D(alpha=0.0001, beta=0.75, n=5, name='norm2')(C)

    #Conv3, Relu3
    C = layers.Conv2D(filters=384,
                      kernel_size=(3,3),
                      padding='same',
                      activation='relu',
                      use_bias=True,
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                      bias_initializer=initializers.Constant(value=0),
                      kernel_regularizer=regularizers.l2(weight_decay),
                      bias_regularizer=None,
                      kernel_constraint=constraints.max_norm(2.),
                      bias_constraint=None,
                      name='conv3')(C)

    #Conv4, Relu4
    grouping = 2
    prev_filters = 384
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[: ,: ,: , i*step : (i+1)*step])(C)
        name = 'conv4_'+str(i)
        groups.append(layers.Conv2D(filters=384//grouping,
                                    kernel_size=(3,3),
                                    padding='same', #maybe same 
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    bias_initializer=initializers.Constant(value=1),
                                    kernel_regularizer=regularizers.l2(weight_decay),
                                    bias_regularizer=None,
                                    kernel_constraint=constraints.max_norm(2.),
                                    bias_constraint=None,
                                    name=name)(group))
    D = layers.Concatenate(name='concat4')(groups)

    #Conv5, Relu5
    grouping = 2
    prev_filters = 384
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[: ,: ,: , i*step : (i+1)*step])(D)
        name = 'conv5_'+str(i)
        groups.append(layers.Conv2D(filters=256//grouping,
                                    kernel_size=(3,3),
                                    padding='same', #maybe same 
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    bias_initializer=initializers.Constant(value=1),
                                    kernel_regularizer=regularizers.l2(weight_decay),
                                    bias_regularizer=None,
                                    kernel_constraint=constraints.max_norm(2.),
                                    bias_constraint=None,
                                    name=name)(group))
    E = layers.Concatenate(name='concat5')(groups)

    #Pool5
    E = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid', name='pool5')(E)

    #Flatten
    E = layers.Flatten(name='flatten')(E)


    #FC6, Relu6
    E = layers.Dense(units=4096,
                           activation='relu',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer=initializers.Constant(value=1),
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=constraints.max_norm(2.),
                           bias_constraint=None,
                           name='fc6')(E)

    #Dropout6
    E = layers.Dropout(rate=0.5, noise_shape=None, name='drop6')(E)

    #FC7, Relu7
    E = layers.Dense(units=4096,
                           activation='relu',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer=initializers.Constant(value=1),
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=constraints.max_norm(2.),
                           bias_constraint=None,
                           name='fc7')(E)

    #Dropout7
    E = layers.Dropout(rate=0.5, noise_shape=None, name='drop7')(E)

    #FC8
    E = layers.Dense(units=num_labels,
                           activation='softmax',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer='zeros',
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=constraints.max_norm(2.),
                           bias_constraint=None,
                           name='fc8')(E)

    model = models.Model(A,E)
    return model


################################################################

#parameters to check
lstm_units = 512

def LSTMCaffeDonahueFunctional(seq_input,num_labels):
    #Input
    A = layers.Input(shape=seq_input)
    #### WEIGHT FILLER - KERNEL INITIALIZER ?????
    # weight filler = gaussian std=0.01
    # kernel initializer = default 'glorot_uniform'
    # https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead
    ####

    #Conv1, Relu1
    B = layers.TimeDistributed(layers.Conv2D(filters=96,
                                            kernel_size=(7,7),
                                            strides=(2,2),
                                            padding='valid', #maybe same 
                                            activation='relu',
                                            use_bias=True,
                                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                            bias_initializer=initializers.Constant(value=0.1),
                                            kernel_regularizer=regularizers.l2(weight_decay),
                                            bias_regularizer=None,
                                            # kernel_constraint=constraints.max_norm(2.),
                                            bias_constraint=None,
                                            ),name='conv1')(A)

    #Pool1
    B = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'), name='pool1')(B)


    #### LRN NORMALIZATION ???
    # Edw pi8anws 8a xreiastei na kanw to LocalResponseNormalization, pou 
    # twra exei afaire8ei ap to keras
    # Sto caffe eixe local_size=5, alpha=0.0001, beta=0.75
    ####
    #Norm1
    B = layers.TimeDistributed(LRN2D(alpha=0.0001, beta=0.75, n=5), name='norm1')(B)


    #Conv2, Relu2 (grouped)
    #### GROUPING
    # group (g): we restrict the connectivity of each filter to a subset
    # of the input. Specifically, the input and output channels are
    # separated into g groups, and the ith output group channels will
    # be only connected to the i-th input group channels.
    # group can reduce resource use, while helping to preserve accuracy.
    #
    # https://stackoverflow.com/a/40876685
    #
    # 8elei group=2
    # https://groups.google.com/forum/?utm_medium=email&utm_source=footer#!topic/keras-users/bxPA4_Bda14
    # https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
    ####

    grouping = 2
    prev_filters = 96
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[: ,: ,: ,:, i*step : (i+1)*step])(B)
        name = 'conv2_'+str(i)
        groups.append(layers.TimeDistributed(layers.Conv2D(filters=384//grouping,
                                                          kernel_size=(5,5),
                                                          strides=(2,2),
                                                          padding='valid', #maybe same 
                                                          activation='relu',
                                                          use_bias=True,
                                                          kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                                          bias_initializer=initializers.Constant(value=0.1),
                                                          kernel_regularizer=regularizers.l2(weight_decay),
                                                          bias_regularizer=None,
                                                          # kernel_constraint=constraints.max_norm(2.),
                                                          bias_constraint=None,),
                                                          name=name)(group))
    C = layers.Concatenate(name='concat2')(groups)

    #Pool2
    C = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same'), name='pool2')(C)

    #Norm2
    C = layers.TimeDistributed(LRN2D(alpha=0.0001, beta=0.75, n=5), name='norm2')(C)

    #Conv3, Relu3
    C = layers.TimeDistributed(layers.Conv2D(filters=512,
                                              kernel_size=(3,3),
                                              padding='same',
                                              activation='relu',
                                              use_bias=True,
                                              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                              bias_initializer=initializers.Constant(value=0.1),
                                              kernel_regularizer=regularizers.l2(weight_decay),
                                              bias_regularizer=None,
                                              # kernel_constraint=constraints.max_norm(2.),
                                              bias_constraint=None),
                                              name='conv3')(C)

    #Conv4, Relu4
    grouping = 2
    prev_filters = 512
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[: ,: ,: ,: , i*step : (i+1)*step])(C)
        name = 'conv4_'+str(i)
        groups.append(layers.TimeDistributed(layers.Conv2D(filters=512//grouping,
                                                            kernel_size=(3,3),
                                                            padding='same', #maybe same 
                                                            activation='relu',
                                                            use_bias=True,
                                                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                                            bias_initializer=initializers.Constant(value=0.1),
                                                            kernel_regularizer=regularizers.l2(weight_decay),
                                                            bias_regularizer=None,
                                                            # kernel_constraint=constraints.max_norm(2.),
                                                            bias_constraint=None),
                                                            name=name)(group))
    D = layers.Concatenate(name='concat4')(groups)

    #Conv5, Relu5
    grouping = 2
    prev_filters = 512
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[: ,: ,:, :, i*step : (i+1)*step])(D)
        name = 'conv5_'+str(i)
        groups.append(layers.TimeDistributed(layers.Conv2D(filters=384//grouping,
                                                            kernel_size=(3,3),
                                                            padding='same', #maybe same 
                                                            activation='relu',
                                                            use_bias=True,
                                                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                                            bias_initializer=initializers.Constant(value=0.1),
                                                            kernel_regularizer=regularizers.l2(weight_decay),
                                                            bias_regularizer=None,
                                                            # kernel_constraint=constraints.max_norm(2.),
                                                            bias_constraint=None),
                                                            name=name)(group))
    E = layers.Concatenate(name='concat5')(groups)

    #Pool5
    E = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='valid'), name='pool5')(E)

    #Flatten
    E = layers.TimeDistributed(layers.Flatten(),name='flatten')(E)


    #FC6, Relu6
    E = layers.TimeDistributed(layers.Dense(units=4096,
                                           activation='relu',
                                           use_bias=True,
                                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                           bias_initializer=initializers.Constant(value=0.1),
                                           kernel_regularizer=regularizers.l2(weight_decay),
                                           bias_regularizer=None,
                                           # kernel_constraint=constraints.max_norm(2.),
                                           bias_constraint=None),
                                           name='fc6')(E)
    #Dropout6
    E = layers.TimeDistributed(layers.Dropout(rate=0.8, noise_shape=None), name='drop6')(E)

    #LSTM

    E = layers.CuDNNLSTM(units=lstm_units,
                        kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01), 
                        recurrent_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01), 
                        bias_initializer='zeros',
                        unit_forget_bias=True,
                        kernel_regularizer=regularizers.l2(weight_decay),
                        recurrent_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        recurrent_constraint=None,
                        bias_constraint=None,
                        return_sequences=False,
                        return_state=False,
                        stateful=False,
                        name='lstm')(E)

        # E = layers.LSTM(units=lstm_units, 
        #             activation='tanh', 
        #             recurrent_activation='hard_sigmoid', 
        #             use_bias=True, 
        #             kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01), 
        #             recurrent_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01), 
        #             bias_initializer='zeros', 
        #             unit_forget_bias=True, 
        #             kernel_regularizer=regularizers.l2(weight_decay),
        #             recurrent_regularizer=None, 
        #             bias_regularizer=None,
        #             # kernel_constraint=constraints.max_norm(2.),
        #             activity_regularizer=None, 
        #             recurrent_constraint=None, 
        #             bias_constraint=None, 
        #             dropout=0.5, 
        #             recurrent_dropout=0.5, 
        #             implementation=2, 
        #             return_sequences=False, 
        #             return_state=False, 
        #             go_backwards=False, 
        #             stateful=False, 
        #             unroll=False,
        #             name='lstm')(E)

    #FC8
    E = layers.Dense(units=num_labels,
                     activation='softmax',
                     use_bias=True,
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                     bias_initializer='zeros',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     bias_regularizer=None,
                     # kernel_constraint=constraints.max_norm(2.),
                     bias_constraint=None,
                     name='fc8-ucf-new')(E)

    model = models.Model(A,E)
    return model





#fullarei h mnhmh
def C3D(seq_input,num_labels):
    """ Return the Keras model of the network
    """
    model = models.Sequential()
    # 1st layer group
    model.add(layers.Conv3D(64, kernel_size=(1, 3, 3), activation='relu', 
                            padding='same', name='conv1',
                            strides=(1, 1, 1), 
                            input_shape=seq_input))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(layers.Conv3D(128, kernel_size=(1, 3, 3), activation='relu', 
                            padding='same', name='conv2',
                            strides=(1, 1, 1)))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(layers.Conv3D(256, kernel_size=(1, 3, 3), activation='relu', 
                            padding='same', name='conv3a',
                            strides=(1, 1, 1)))
    model.add(layers.Conv3D(256, kernel_size=(1, 3, 3), activation='relu', 
                            padding='same', name='conv3b',
                            strides=(1, 1, 1)))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(layers.Conv3D(512, kernel_size=(1, 3, 3), activation='relu', 
                            padding='same', name='conv4a',
                            strides=(1, 1, 1)))
    model.add(layers.Conv3D(512, kernel_size=(1, 3, 3), activation='relu', 
                            padding='same', name='conv4b',
                            strides=(1, 1, 1)))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(layers.Conv3D(512, kernel_size=(1, 3, 3), activation='relu', 
                            padding='same', name='conv5a',
                            strides=(1, 1, 1)))
    model.add(layers.Conv3D(512, kernel_size=(1, 3, 3), activation='relu', 
                            padding='same', name='conv5b',
                            strides=(1, 1, 1)))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           padding='valid', name='pool5'))
    model.add(layers.Flatten())
    # FC layers group
    model.add(layers.Dense(4096, activation='relu', name='fc6'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(4096, activation='relu', name='fc7'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(num_labels, activation='softmax', name='fc8'))

    return model


#fullarei h mnhmh
def C3D_custom(seq_input, num_labels):
    
    #Input
    A = layers.Input(shape=seq_input)
    #### WEIGHT FILLER - KERNEL INITIALIZER ?????
    # weight filler = gaussian std=0.01
    # kernel initializer = default 'glorot_uniform'
    # https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead
    ####

    #Conv1, Relu1
    B = layers.Conv3D(filters=96,
                      kernel_size=(1,7,7),
                      strides=(1,2,2),
                      padding='valid', #maybe same 
                      activation='relu',
                      use_bias=True,
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                      bias_initializer=initializers.Constant(value=0.1),
                      kernel_regularizer=regularizers.l2(weight_decay),
                      bias_regularizer=None,
                      kernel_constraint=constraints.max_norm(2.),
                      bias_constraint=None,
                      name='conv1')(A)

    #Pool1
    B = layers.MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2), padding='valid', name='pool1')(B)


    #### LRN NORMALIZATION ???
    # Edw pi8anws 8a xreiastei na kanw to LocalResponseNormalization, pou 
    # twra exei afaire8ei ap to keras
    # Sto caffe eixe local_size=5, alpha=0.0001, beta=0.75
    ####
    #Norm1
    B = LRN2D(alpha=0.0001, beta=0.75, n=5, name='norm1')(B)

    #Conv2, Relu2 (grouped)
    #### GROUPING
    # group (g): we restrict the connectivity of each filter to a subset
    # of the input. Specifically, the input and output channels are
    # separated into g groups, and the ith output group channels will
    # be only connected to the i-th input group channels.
    # group can reduce resource use, while helping to preserve accuracy.
    #
    # https://stackoverflow.com/a/40876685
    #
    # 8elei group=2
    # https://groups.google.com/forum/?utm_medium=email&utm_source=footer#!topic/keras-users/bxPA4_Bda14
    # https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
    ####

    grouping = 2
    prev_filters = 96
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[:, : ,: ,: , i*step : (i+1)*step])(B)
        name = 'conv2_'+str(i)
        groups.append(layers.Conv3D(filters=384//grouping,
                                    kernel_size=(1,5,5),
                                    strides=(1,2,2),
                                    padding='valid', #maybe same 
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    bias_initializer=initializers.Constant(value=0.1),
                                    kernel_regularizer=regularizers.l2(weight_decay),
                                    bias_regularizer=None,
                                    kernel_constraint=constraints.max_norm(2.),
                                    bias_constraint=None,
                                    name=name)(group))
    C = layers.Concatenate(name='concat2')(groups)

    #Pool2
    C = layers.MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2), padding='same', name='pool2')(C)

    #Norm2
    C = LRN2D(alpha=0.0001, beta=0.75, n=5, name='norm2')(C)

    #Conv3, Relu3
    C = layers.Conv3D(filters=512,
                      kernel_size=(1,3,3),
                      padding='same',
                      activation='relu',
                      use_bias=True,
                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                      bias_initializer=initializers.Constant(value=0.1),
                      kernel_regularizer=regularizers.l2(weight_decay),
                      bias_regularizer=None,
                      kernel_constraint=constraints.max_norm(2.),
                      bias_constraint=None,
                      name='conv3')(C)

    #Conv4, Relu4
    grouping = 2
    prev_filters = 512
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[: ,: , : ,: , i*step : (i+1)*step])(C)
        name = 'conv4_'+str(i)
        groups.append(layers.Conv3D(filters=512//grouping,
                                    kernel_size=(1,3,3),
                                    padding='same', #maybe same 
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    bias_initializer=initializers.Constant(value=0.1),
                                    kernel_regularizer=regularizers.l2(weight_decay),
                                    bias_regularizer=None,
                                    kernel_constraint=constraints.max_norm(2.),
                                    bias_constraint=None,
                                    name=name)(group))
    D = layers.Concatenate(name='concat4')(groups)

    #Conv5, Relu5
    grouping = 2
    prev_filters = 512
    step = prev_filters//grouping

    groups = []
    for i in range(grouping):
        group = layers.Lambda(lambda z: z[:, : ,: ,: , i*step : (i+1)*step])(D)
        name = 'conv5_'+str(i)
        groups.append(layers.Conv3D(filters=384//grouping,
                                    kernel_size=(1,3,3),
                                    padding='same', #maybe same 
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    bias_initializer=initializers.Constant(value=0.1),
                                    kernel_regularizer=regularizers.l2(weight_decay),
                                    bias_regularizer=None,
                                    kernel_constraint=constraints.max_norm(2.),
                                    bias_constraint=None,
                                    name=name)(group))
    E = layers.Concatenate(name='concat5')(groups)

    #Pool5
    E = layers.MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2), padding='valid', name='pool5')(E)

    #Flatten
    E = layers.Flatten(name='flatten')(E)


    #FC6, Relu6
    E = layers.Dense(units=4096,
                           activation='relu',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer=initializers.Constant(value=0.1),
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=constraints.max_norm(2.),
                           bias_constraint=None,
                           name='fc6')(E)

    #Dropout6
    E = layers.Dropout(rate=0.5, noise_shape=None, name='drop6')(E)

    #FC7, Relu7
    E = layers.Dense(units=4096,
                           activation='relu',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer=initializers.Constant(value=0.1),
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=constraints.max_norm(2.),
                           bias_constraint=None,
                           name='fc7')(E)

    #Dropout7
    E = layers.Dropout(rate=0.5, noise_shape=None, name='drop7')(E)

    #FC8
    E = layers.Dense(units=num_labels,
                           activation='softmax',
                           use_bias=True,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                           bias_initializer='zeros',
                           kernel_regularizer=regularizers.l2(weight_decay),
                           bias_regularizer=None,
                           kernel_constraint=constraints.max_norm(2.),
                           bias_constraint=None,
                           name='fc8-ucf')(E)

    model = models.Model(A,E)
    return model




