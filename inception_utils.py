from keras.applications.inception_v3 import InceptionV3
from keras import layers
from keras import models
from keras import optimizers



def get_model(weights,input_shape,num_labels):
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False,input_shape=input_shape)

    # add a global spatial average pooling layer
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = layers.Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = layers.Dense(num_labels, activation='softmax')(x)

    # this is the model we will train
    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model


def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def freeze_all_but_mid_and_top(model,base_lr,momentum):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=optimizers.SGD(lr=base_lr, momentum=momentum),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model