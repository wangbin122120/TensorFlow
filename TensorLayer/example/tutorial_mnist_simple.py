#! /usr/bin/python
# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorlayer as tl

sess = tf.InteractiveSession()

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = \
                                tl.files.load_mnist_dataset(path="/tmp/TensorFlow/Tensorlayer/input/mnist/",shape=(-1,784))
# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# define the network
network = tl.layers.InputLayer(x, name='input')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
# the softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
# speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()
network = tl.layers.DenseLayer(network, n_units=10,
                                act=tf.identity, name='output')

# define cost function and metric.
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, name='cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# print network information
network.print_params()
network.print_layers()

# train the network
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=500, n_epoch=500, print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False)

# evaluation
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

# save the network to .npz file
tl.files.save_npz(network.all_params , name='/tmp/TensorFlow/Tensorlayer/model/mnist_simple.npz')
sess.close()



'''
Epoch 495 of 500 took 0.384348s
   val loss: 0.055421
   val acc: 0.986100
Epoch 500 of 500 took 0.446057s
   val loss: 0.056253
   val acc: 0.986000
Total training time: 204.117375s
Start testing the network ...
   test loss: 0.049387
   test acc: 0.986400
[*] model.npz saved

-----------------------------------------

# 40%  1080ti@1.9GHz use 900 MB, 30% E5-1620-v4@3.5GHz,  19'48" -  23'19"
-----------------------------------------
C:\Anaconda3\python.exe C:/Users/w/project/git/TensorFlow/TensorLayer/example/tutorial_mnist_simple.py
2017-09-14 01:11:40.813765: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-09-14 01:11:40.814032: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-14 01:11:41.198790: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.6325
pciBusID 0000:03:00.0
Total memory: 11.00GiB
Free memory: 9.08GiB
2017-09-14 01:11:41.199096: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:976] DMA: 0 
2017-09-14 01:11:41.199238: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:986] 0:   Y 
2017-09-14 01:11:41.199399: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0)
Load or Download MNIST > /tmp/TensorFlow/Tensorlayer/data/mnist/
Downloading train-images-idx3-ubyte.gz...100%
Succesfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
/tmp/TensorFlow/Tensorlayer/data/mnist/train-images-idx3-ubyte.gz
Downloading train-labels-idx1-ubyte.gz...113%
Succesfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Downloading t10k-images-idx3-ubyte.gz...100%
Succesfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
/tmp/TensorFlow/Tensorlayer/data/mnist/t10k-images-idx3-ubyte.gz
Downloading t10k-labels-idx1-ubyte.gz...180%
Succesfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
  [TL] InputLayer  input: (?, 784)
  [TL] DropoutLayer drop1: keep:0.800000 is_fix:False
  [TL] DenseLayer  relu1: 800 relu
  [TL] DropoutLayer drop2: keep:0.500000 is_fix:False
  [TL] DenseLayer  relu2: 800 relu
  [TL] DropoutLayer drop3: keep:0.500000 is_fix:False
  [TL] DenseLayer  output: 10 identity
  param   0: relu1/W:0            (784, 800)         float32_ref (mean: -0.0002222577459178865, median: -0.0002490733750164509, std: 0.08795216679573059)   
  param   1: relu1/b:0            (800,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )   
  param   2: relu2/W:0            (800, 800)         float32_ref (mean: -7.916052709333599e-05, median: -2.623737236717716e-05, std: 0.0880211740732193)   
  param   3: relu2/b:0            (800,)             float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )   
  param   4: output/W:0           (800, 10)          float32_ref (mean: -0.0004766570054925978, median: -0.0012888314668089151, std: 0.08838297426700592)   
  param   5: output/b:0           (10,)              float32_ref (mean: 0.0               , median: 0.0               , std: 0.0               )   
  num of params: 1276810
  layer   0: drop1/mul:0          (?, 784)           float32
  layer   1: relu1/Relu:0         (?, 800)           float32
  layer   2: drop2/mul:0          (?, 800)           float32
  layer   3: relu2/Relu:0         (?, 800)           float32
  layer   4: drop3/mul:0          (?, 800)           float32
  layer   5: output/Identity:0    (?, 10)            float32
Start training the network ...
Epoch 1 of 500 took 0.738969s
   val loss: 0.614407
   val acc: 0.803200
Epoch 5 of 500 took 0.393881s
   val loss: 0.293534
   val acc: 0.912100
Epoch 10 of 500 took 0.398998s
   val loss: 0.227673
   val acc: 0.935600
Epoch 15 of 500 took 0.402516s
   val loss: 0.189769
   val acc: 0.949400
Epoch 20 of 500 took 0.398497s
   val loss: 0.165832
   val acc: 0.956400
Epoch 25 of 500 took 0.458571s
   val loss: 0.146433
   val acc: 0.961200
Epoch 30 of 500 took 0.463019s
   val loss: 0.132032
   val acc: 0.964100
Epoch 35 of 500 took 0.396495s
   val loss: 0.120098
   val acc: 0.967600
Epoch 40 of 500 took 0.400499s
   val loss: 0.111306
   val acc: 0.969500
Epoch 45 of 500 took 0.396071s
   val loss: 0.103029
   val acc: 0.971500
Epoch 50 of 500 took 0.390987s
   val loss: 0.097054
   val acc: 0.973100
Epoch 55 of 500 took 0.409010s
   val loss: 0.092964
   val acc: 0.974900
Epoch 60 of 500 took 0.399547s
   val loss: 0.087990
   val acc: 0.976000
Epoch 65 of 500 took 0.398949s
   val loss: 0.084225
   val acc: 0.976500
Epoch 70 of 500 took 0.403126s
   val loss: 0.079975
   val acc: 0.977800
Epoch 75 of 500 took 0.398998s
   val loss: 0.077527
   val acc: 0.978600
Epoch 80 of 500 took 0.394482s
   val loss: 0.075330
   val acc: 0.978700
Epoch 85 of 500 took 0.399498s
   val loss: 0.072437
   val acc: 0.979400
Epoch 90 of 500 took 0.396409s
   val loss: 0.070494
   val acc: 0.979500
Epoch 95 of 500 took 0.396529s
   val loss: 0.069358
   val acc: 0.980400
Epoch 100 of 500 took 0.394510s
   val loss: 0.067484
   val acc: 0.980400
Epoch 105 of 500 took 0.397540s
   val loss: 0.066551
   val acc: 0.981200
Epoch 110 of 500 took 0.395625s
   val loss: 0.064961
   val acc: 0.981300
Epoch 115 of 500 took 0.391090s
   val loss: 0.063403
   val acc: 0.981700
Epoch 120 of 500 took 0.392491s
   val loss: 0.062178
   val acc: 0.982300
Epoch 125 of 500 took 0.397495s
   val loss: 0.062243
   val acc: 0.982000
Epoch 130 of 500 took 0.388985s
   val loss: 0.060891
   val acc: 0.982800
Epoch 135 of 500 took 0.393020s
   val loss: 0.059996
   val acc: 0.982500
Epoch 140 of 500 took 0.394463s
   val loss: 0.059287
   val acc: 0.983200
Epoch 145 of 500 took 0.395653s
   val loss: 0.058832
   val acc: 0.983400
Epoch 150 of 500 took 0.393023s
   val loss: 0.057920
   val acc: 0.983400
Epoch 155 of 500 took 0.399855s
   val loss: 0.057216
   val acc: 0.983000
Epoch 160 of 500 took 0.391159s
   val loss: 0.056804
   val acc: 0.983200
Epoch 165 of 500 took 0.393039s
   val loss: 0.056979
   val acc: 0.983400
Epoch 170 of 500 took 0.433041s
   val loss: 0.056727
   val acc: 0.983900
Epoch 175 of 500 took 0.430059s
   val loss: 0.056342
   val acc: 0.984500
Epoch 180 of 500 took 0.389509s
   val loss: 0.056092
   val acc: 0.984600
Epoch 185 of 500 took 0.392059s
   val loss: 0.055347
   val acc: 0.984600
Epoch 190 of 500 took 0.392040s
   val loss: 0.055569
   val acc: 0.984500
Epoch 195 of 500 took 0.389498s
   val loss: 0.055073
   val acc: 0.984800
Epoch 200 of 500 took 0.386841s
   val loss: 0.055577
   val acc: 0.984400
Epoch 205 of 500 took 0.392015s
   val loss: 0.055027
   val acc: 0.984100
Epoch 210 of 500 took 0.397067s
   val loss: 0.055087
   val acc: 0.984700
Epoch 215 of 500 took 0.390963s
   val loss: 0.054938
   val acc: 0.984500
Epoch 220 of 500 took 0.389466s
   val loss: 0.055018
   val acc: 0.985000
Epoch 225 of 500 took 0.391491s
   val loss: 0.054695
   val acc: 0.984500
Epoch 230 of 500 took 0.389345s
   val loss: 0.054605
   val acc: 0.984600
Epoch 235 of 500 took 0.392530s
   val loss: 0.053990
   val acc: 0.985400
Epoch 240 of 500 took 0.396159s
   val loss: 0.054128
   val acc: 0.984600
Epoch 245 of 500 took 0.396994s
   val loss: 0.055122
   val acc: 0.984700
Epoch 250 of 500 took 0.391518s
   val loss: 0.054374
   val acc: 0.984900
Epoch 255 of 500 took 0.386966s
   val loss: 0.053582
   val acc: 0.985200
Epoch 260 of 500 took 0.384983s
   val loss: 0.054493
   val acc: 0.984700
Epoch 265 of 500 took 0.386112s
   val loss: 0.054306
   val acc: 0.984600
Epoch 270 of 500 took 0.389227s
   val loss: 0.053486
   val acc: 0.985600
Epoch 275 of 500 took 0.385867s
   val loss: 0.053676
   val acc: 0.985700
Epoch 280 of 500 took 0.390585s
   val loss: 0.054458
   val acc: 0.985000
Epoch 285 of 500 took 0.394917s
   val loss: 0.053817
   val acc: 0.985100
Epoch 290 of 500 took 0.387983s
   val loss: 0.054568
   val acc: 0.985200
Epoch 295 of 500 took 0.387027s
   val loss: 0.055273
   val acc: 0.985900
Epoch 300 of 500 took 0.390257s
   val loss: 0.055213
   val acc: 0.985800
Epoch 305 of 500 took 0.395566s
   val loss: 0.055450
   val acc: 0.985400
Epoch 310 of 500 took 0.391006s
   val loss: 0.054991
   val acc: 0.985700
Epoch 315 of 500 took 0.385973s
   val loss: 0.055225
   val acc: 0.986100
Epoch 320 of 500 took 0.391544s
   val loss: 0.055121
   val acc: 0.985600
Epoch 325 of 500 took 0.386481s
   val loss: 0.054824
   val acc: 0.984900
Epoch 330 of 500 took 0.391474s
   val loss: 0.054893
   val acc: 0.985600
Epoch 335 of 500 took 0.394096s
   val loss: 0.055630
   val acc: 0.985600
Epoch 340 of 500 took 0.389039s
   val loss: 0.055014
   val acc: 0.985900
Epoch 345 of 500 took 0.387106s
   val loss: 0.056781
   val acc: 0.985100
Epoch 350 of 500 took 0.388292s
   val loss: 0.056318
   val acc: 0.985500
Epoch 355 of 500 took 0.390488s
   val loss: 0.055544
   val acc: 0.984900
Epoch 360 of 500 took 0.384057s
   val loss: 0.057808
   val acc: 0.985500
Epoch 365 of 500 took 0.389602s
   val loss: 0.055144
   val acc: 0.985700
Epoch 370 of 500 took 0.387317s
   val loss: 0.054797
   val acc: 0.985800
Epoch 375 of 500 took 0.393040s
   val loss: 0.057173
   val acc: 0.984600
Epoch 380 of 500 took 0.398516s
   val loss: 0.054702
   val acc: 0.986300
Epoch 385 of 500 took 0.393481s
   val loss: 0.055328
   val acc: 0.986300
Epoch 390 of 500 took 0.399088s
   val loss: 0.054736
   val acc: 0.985400
Epoch 395 of 500 took 0.394992s
   val loss: 0.055112
   val acc: 0.986100
Epoch 400 of 500 took 0.408853s
   val loss: 0.056453
   val acc: 0.986000
Epoch 405 of 500 took 0.399571s
   val loss: 0.054022
   val acc: 0.986500
Epoch 410 of 500 took 0.409009s
   val loss: 0.054304
   val acc: 0.985800
Epoch 415 of 500 took 0.390012s
   val loss: 0.055299
   val acc: 0.985500
Epoch 420 of 500 took 0.389986s
   val loss: 0.055357
   val acc: 0.986300
Epoch 425 of 500 took 0.392044s
   val loss: 0.055704
   val acc: 0.985600
Epoch 430 of 500 took 0.386482s
   val loss: 0.055844
   val acc: 0.985200
Epoch 435 of 500 took 0.391034s
   val loss: 0.056036
   val acc: 0.985700
Epoch 440 of 500 took 0.390500s
   val loss: 0.054864
   val acc: 0.985600
Epoch 445 of 500 took 0.393574s
   val loss: 0.055034
   val acc: 0.985900
Epoch 450 of 500 took 0.385389s
   val loss: 0.054707
   val acc: 0.985900
Epoch 455 of 500 took 0.392799s
   val loss: 0.054327
   val acc: 0.986200
Epoch 460 of 500 took 0.386485s
   val loss: 0.054667
   val acc: 0.986000
Epoch 465 of 500 took 0.471945s
   val loss: 0.054721
   val acc: 0.986300
Epoch 470 of 500 took 0.386962s
   val loss: 0.055266
   val acc: 0.986000
Epoch 475 of 500 took 0.386504s
   val loss: 0.055077
   val acc: 0.986300
Epoch 480 of 500 took 0.384500s
   val loss: 0.053925
   val acc: 0.986000
Epoch 485 of 500 took 0.387495s
   val loss: 0.054757
   val acc: 0.986200
Epoch 490 of 500 took 0.387113s
   val loss: 0.054920
   val acc: 0.986000
Epoch 495 of 500 took 0.384348s
   val loss: 0.055421
   val acc: 0.986100
Epoch 500 of 500 took 0.446057s
   val loss: 0.056253
   val acc: 0.986000
Total training time: 204.117375s
Start testing the network ...
   test loss: 0.049387
   test acc: 0.986400
[*] model.npz saved


'''