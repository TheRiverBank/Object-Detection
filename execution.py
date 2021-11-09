import cv2
import matplotlib.pyplot as plt

from model import Model
import numpy as np
from dataProcessing import resize_img

labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def run_test(model):

    img = cv2.imread('./images/dog_test.PNG')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = resize_img(img)
    plt.imshow(img_resized[0,:,:,:])
    plt.show()
    prediction = model.predict_proba(img_resized)
    print(prediction)
    return labels[np.argmax(prediction)]


m = Model()
cnn = m.get_model()
prediction = run_test(cnn)
print(prediction)