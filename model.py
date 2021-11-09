import sys
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.constraints import max_norm
from matplotlib import pyplot as plt
from dataProcessing import load_data

class Model:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.classes = 10

    def get_model(self):


        model = load_model('./models/model')


        return model

    def create_model(self):
        X_train, Y_train, x_test, y_test = load_data()

        Y_train = np_utils.to_categorical(Y_train, self.classes)
        y_test = np_utils.to_categorical(y_test, self.classes)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(32,32,3)))
        model.add(Dropout(0.20))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.20))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.20))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_constraint=max_norm(3)))
        model.add(Dropout(0.30))
        model.add(Dense(self.classes, activation='softmax'))

        epochs = 1

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train, Y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64)

        self.summarize_diagnostics(history)

        model.save('./models/model')

    def summarize_diagnostics(self, history):
        plt.subplot(121)
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        plt.title('Loss function')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(122)
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        filename = sys.argv[0].split('/')[-1]
        plt.savefig(filename + '_plot.png')
        plt.show()
        plt.close()

m = Model()
m.create_model()
