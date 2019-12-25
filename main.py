import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import image

x=[]
x_pic = image.imread('befor background removed.png')
x.append(x_pic[:400,:320,:])
x = np.array(x)

y=[]
y_pic = np.array(image.imread('background removed.png'))
y.append(y_pic[:400,:320,:])
y = np.array(y)

plt.figure()
plt.imshow(x[0])

model = tf.keras.Sequential([
                                     tf.keras.layers.Conv2D(320, (2, 2), activation='relu',input_shape=(400,320,3)),
                                     tf.keras.layers.MaxPooling2D((4, 4)),
                                     #encoding layers
                                     tf.keras.layers.Conv2D(320,(1,1),activation="sigmoid"),
                                     #bottleneck layer
                                     tf.keras.layers.UpSampling2D((5, 5)),
                                     tf.keras.layers.Conv2D(3,(96,76),activation="sigmoid")
                                     #decoding layers
])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.3), loss = "categorical_crossentropy")
model.fit(x,y,epochs=1)

