import model
import dataProcessing
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

X_train, Y_train, _, _ = dataProcessing.load_data()

def check_images():
    labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    (X_shuff, Y_shuff) = shuffle(X_train, Y_train)
    rows, cols = 8, 8
    fig, axes = plt.subplots(rows, cols, figsize=(15,15))
    axes = axes.ravel()
    for i in range(0, rows*cols):
        axes[i].imshow(X_shuff[i])
        axes[i].set_title("{}". format(labels[Y_shuff.item(i)]))
        axes[i].axis('off')
        plt.subplots_adjust(wspace=1)

    return

check_images()