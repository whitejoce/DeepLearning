import random

import matplotlib.pyplot as plt
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
window.setWindowTitle('MNIST')

button=QPushButton(window)
button.setText('下一组')
#按钮居中
button.move(360,460)
button.clicked.connect(lambda:show_mnist(model,x_test,y_test))

def train_mnist():
    #可视化数据
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i],cmap=plt.cm.binary)
        plt.xlabel(y_train[i])
    plt.show()
    #构建模型
    from keras.models import Sequential
    from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
    
    model=Sequential()
    #卷积层
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    #全连接层
    model.add(Dense(512,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.summary()
    #编译模型
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #训练模型
    model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
    #保存模型
    model.save('mnist.h5')

def show_mnist(model,x_test,y_test):
    pos=random.randint(0,10000-11)
    j=0
    for i in range(pos,pos+10):
        plt.subplot(2,5,j+1)
        j+=1
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i],cmap=plt.cm.binary)
        plt.xlabel("real:"+str(y_test[i])+'\npredict:'+str(np.argmax(model.predict(x_test[i].reshape(1,28,28,1)))))
    canvas.draw()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train=x_train.reshape(x_train.shape[0],28,28,1)/255
    x_test=x_test.reshape(x_test.shape[0],28,28,1)/255
    print(x_test.shape)
    #训练模型
    train_mnist()
    #加载模型
    model=load_model('mnist.h5')
    #重新加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    show_mnist(model,x_test,y_test)
    
window.show()
app.exec_()
