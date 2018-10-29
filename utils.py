import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
from custom_models import *


# https://ksaluja15.github.io/Learning-Rate-Multipliers-in-Keras/
from LR_SGD import *
import math
####################################
def model_compile(model,base_lr, momentum):
    ####  MULTIPLIERS
    # ston solver einai base_lr = 0.001 kai weight_decay=0.005
    # auta pollaplasiazontai me ta lr_mult kai decay_mult kai prokuptei gia
    # ka8e layer diaforetiko. px edw einai:
    # w_lr_mult=1
    # w_decay_mult=1
    # b_lr_mult=2
    # b_decay_mult=0
    # (pantou etsi einai, ektos ap to ucf8 layer, pou einai 10,20 ta lr)
    #
    # Set the weight learning rate to be the same as the learning rate given 
    # by the solver during runtime, and the bias learning rate to be twice
    # as large as that - this usually leads to better convergence rates.
    #
    # Possible solution : https://ksaluja15.github.io/Learning-Rate-Multipliers-in-Keras/
    ####


    # Setting the Learning rate multipliers
    LR_mult_dict = {}
    LR_mult_dict['conv1']=1
    LR_mult_dict['conv1']=1
    LR_mult_dict['conv1']=1
    LR_mult_dict['conv1']=1
    LR_mult_dict['conv1']=1
    LR_mult_dict['fc6']=1
    LR_mult_dict['fc7']=1
    LR_mult_dict['fc8-ucf']=10
    LR_mult_dict['fc8-ucf-new']=10


    # Setting up custom optimizer by 
    optimizer = LR_SGD(lr=base_lr,
                       momentum=momentum,
                       decay=0.0,
                       nesterov=True,
                       multipliers = LR_mult_dict)

    # Sto Caffe, xrhsimopoiei ws loss function thn SoftmaxWithLoss
    # pou emperiexei thn MultinomialLogisticLoss. 
    # Crossentropy is applied here, they are the same
    # https://stats.stackexchange.com/questions/166958/multinomial-logistic-loss-vs-cross-entropy-vs-square-error
    #
    # Multiclass, single-label classification
    # Activation: softmax
    # Loss:  categorical_crossentropy, by Cholet, p. 114
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print("{}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()

def get_h5_weights(weight_file_path):
    f = h5py.File(weight_file_path)
    # if len(f.attrs.items()):
    #     print("{} contains: ".format(weight_file_path))
    #     print("Root attributes:")
    # for key, value in f.attrs.items():
    #     print("  {}: {}".format(key, value))
    # print('\n\n')

    dictionary_of_weights = {}
    for layer, g in f.items():
    #     print("  {}".format(layer))
    #     for key, value in g.attrs.items():
    #         print("      {}: {}".format(key, value))

        for p_name in g.keys():
            param = g[p_name]
            subkeys = param.keys()
            for k_name in param.keys():
                # get rid of annoying :0 at the end
                k_name2 = k_name.split(":0")[0]
    #             print("{}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
                string = p_name+'/'+k_name2
                dictionary_of_weights[string] = param.get(k_name)[:]
    f.close()

    # for key,item in zip(dictionary_of_weights.keys(),dictionary_of_weights.items()):
    #     print("{}\t{}".format(key,item[1].shape))
    return dictionary_of_weights

def load_pretrained_weights(model, pm_dict):
    grouping = 2 # this value for grouping convolutions was used to create the model architecture

    for i in range(len(model.layers)):
        curr_layer = model.layers[i]
        curr_weights = curr_layer.get_weights()

        # if current layer has kernel and bias (is trainable...?)
        if(len(curr_weights)!=0):
            curr_name = curr_layer.name
            layername = curr_name.split('_')[0]
            # do not do anything for the last layer, as it maybe has 
            # different dimensions than your model
            # if('fc8' in layername):
            #     continue

            # if grouped split pretrained weights between _ind convolutions
            bias = pm_dict[layername+'/bias']
            kernel = pm_dict[layername+'/kernel']
            if ('conv' in curr_name) and ('_' in curr_name) :
                ind = int(curr_name.split('_')[1])
                channels = kernel.shape[3]
                step = channels//grouping
                bias = bias[ind*step : (ind+1)*step]
                kernel = kernel[:,:,:,ind*step : (ind+1)*step]        

            # else simply set weights    
            lista = [kernel, bias]
            model.layers[i].set_weights(lista)    
    return model

def load_pretrained_weights2(model, arrays_path):
    onlyfiles = [f for f in listdir(arrays_path) if isfile(join(arrays_path, f))]
    # print(onlyfiles)

    tot_layers={}
    for i in onlyfiles:
        tot_layers[i.split('.npy')[0]] = np.load(arrays_path+i)
    # print(tot_layers)

    grouping = 2 # this value for grouping convolutions was used to create the model architecture

    for i in range(len(model.layers)):
        curr_layer = model.layers[i]
        curr_weights = curr_layer.get_weights()

        # if current layer has kernel and bias (is trainable...?)
        if(len(curr_weights)!=0):
            curr_name = curr_layer.name
            layername = curr_name.split('_')[0]
            # do not do anything for the last layer, as it maybe has 
            # different dimensions than your model
            # if('fc8' in layername):
            #     continue

            # if grouped split pretrained weights between _ind convolutions
            kernel = tot_layers[layername+'-kernel']
            bias = tot_layers[layername+'-bias']
            if ('conv' in curr_name) and ('_' in curr_name) :
                ind = int(curr_name.split('_')[1])
                channels = kernel.shape[3]
                step = channels//grouping
                bias = bias[ind*step : (ind+1)*step]
                kernel = kernel[:,:,:,ind*step : (ind+1)*step]        

            # else simply set weights    
            lista = [kernel, bias]
            model.layers[i].set_weights(lista)
    return model



def load_single_frame_weights(lstm_model,saved_weights):
    input_shape = lstm_model.get_input_shape_at(0)[2:] #it was (None,seq_len, input_shape)
    num_labels = lstm_model.get_output_shape_at(-1)[1] #it was (None, num_labels)
    
    model = CaffeDonahueFunctional(input_shape=input_shape,num_labels=num_labels)
    #remove last 3 layers (fc8-ucf, drop7, fc7)
    model.load_weights(saved_weights)
    for i in range(3):
        model.layers.pop()
        # print('Currently removing : ' + str(model.layers.pop()))  

    #pass single frame weights to the first 25 layers of the network (the layers that are have the same configuration)
    for i in range(len(lstm_model.layers[:26])):
        if(len(lstm_model.layers[i].get_weights())>0): #for those layers that are trainable (have weights)
            layer_name = lstm_model.layers[i].name
            single_frame_weights = model.get_layer(layer_name).get_weights()
            bias = single_frame_weights[1]
            kernel = single_frame_weights[0]
            lista = [kernel, bias]
            lstm_model.layers[i].set_weights(lista)    
    
    return lstm_model

