import random

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

app = QApplication([])
window = QMainWindow()
window.setGeometry(100, 100, 800, 500)

canvas = FigureCanvas(plt.figure())
window.setCentralWidget(canvas)
window.setWindowTitle('Fashion_MNIST')

button=QPushButton(window)
button.setText('下一组')
#按钮居中
button.move(360,460)
button.clicked.connect(lambda:show_mnist(model,test_images,test_labels))

cloth=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def train_mnist():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    print(train_images.shape, train_labels.shape)

    #plt.imshow(train_images[0])
    #plt.show()

    train_images=train_images/255.0
    test_images=test_images/255.0

    #one_hot编码
    #train_labels_onehot=tf.keras.utils.to_categorical(train_labels)
    #test_labels_onehot=tf.keras.utils.to_categorical(test_labels)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    #使用one_hot编码: loss='categorical_crossentropy'
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    #model.fit(train_images, train_labels_onehot, epochs=15)
    model.fit(train_images, train_labels, epochs=15)

    model.save('model.h5')
    

def show_mnist(model,test_images,test_labels):
    pos=random.randint(0,10000-11)
    j=0
    for i in range(pos,pos+10):
        plt.subplot(2,5,j+1)
        j+=1
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i],cmap=plt.cm.binary)
        plt.xlabel("real:"+cloth[test_labels[i]]+'\npredict:'+cloth[np.argmax(model.predict(test_images[i].reshape(1,28,28,1)))])
    canvas.draw()

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images=train_images.reshape(train_images.shape[0],28,28,1)/255
    test_images=test_images.reshape(test_images.shape[0],28,28,1)/255
    print(test_images.shape)
    #训练模型
    train_mnist()
    #加载模型
    model=load_model('model.h5')
    show_mnist(model,test_images,test_labels)
window.show()
app.exec_()
