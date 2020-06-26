''' This module evaluates the performance of a trained CPC encoder '''

from maskdata_utils import FaceMaskGenerator
from os.path import join, basename, dirname, exists
import keras
import tensorflow

config = tensorflow.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16} ) 
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config=config) 
keras.backend.set_session(sess)

def build_model(image_shape, learning_rate):
    # Define the classifier
    x_input = keras.layers.Input(image_shape)
    x = keras.layers.AveragePooling2D((7,7))(x_input)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=128, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    output = keras.layers.Dense(units=2, activation='softmax')(x)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=output)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()

    return model


def embedgenerator(data, encoder):
    while True:
        x, y = next(data)
        result = encoder.predict(x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
        result = result.reshape(x.shape[0], 7, 7, 128)
        yield result, y


def benchmark_model(encoder_path, epochs, batch_size, output_dir, lr=1e-4, color=False):

    # Prepare data
    train_data = FaceMaskGenerator(batch_size, subset='train', train_ratio=1.0, rescale=True)

    validation_data = FaceMaskGenerator(batch_size, subset='valid', rescale=True)

    encoder = keras.models.load_model(encoder_path)

    # Prepares the model
    model = build_model(image_shape=(7, 7, 128), learning_rate=lr)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]

    _ = next(embedgenerator(train_data, encoder)) # bug without this line get error ValueError: Tensor is not an element of this graph.

    # Trains the model
    model.fit_generator(
        generator=embedgenerator(train_data, encoder),
        steps_per_epoch=len(train_data),
        validation_data=embedgenerator(validation_data, encoder),
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # Saves the model
    model.save(join(output_dir, 'supervised.h5'))


if __name__ == "__main__":

    benchmark_model(
        encoder_path='mymodels/facemask/encoder.h5',
        epochs=15,
        batch_size=64,
        output_dir='mymodels/facemask',
        lr=1e-3,
        color=True
    )
