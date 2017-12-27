import numpy as np
import cv2
import os

def conv_mnist(img_file,label_file, num, path, list,label):

    imgs=open(img_file).read()
    imgs=np.fromstring(imgs,dtype=np.uint8)
    imgs=imgs[16:] # magic number -> skip 4 integers
    imgs=imgs.reshape((num,28,28))

    labels=open(label_file).read()
    labels=np.fromstring(labels,dtype=np.uint8)
    labels=labels[8:] # magic number -> skip 2 integers

    fw=open(list,"w")
    for i in range(num):
        class_id = labels[i]
        img_name = "%s_%05d_c%d.png" % (label,i,class_id)
        img_path = path+"/"+img_name
        cv2.imwrite(img_path,imgs[i])
        fw.write(img_path+"\n")
    fw.close()

os.system('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
os.system('wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
os.system('wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
os.system('wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')

os.system('gunzip train-images-idx3-ubyte.gz')
os.system('gunzip train-labels-idx1-ubyte.gz')
os.system('gunzip t10k-images-idx3-ubyte.gz')
os.system('gunzip t10k-labels-idx1-ubyte.gz')

cwd = os.getcwd()
os.system('mkdir train_image')
os.system('mkdir test_image')
img_path=cwd+"/train_image"
conv_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000, img_path, "mnist.train.list","t")

img_path=cwd+"/test_image"
conv_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000, img_path, "mnist.test.list","v")

os.system('rm train-*')
os.system('rm t10k-*')
