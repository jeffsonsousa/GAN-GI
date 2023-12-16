import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
from tensorflow.keras.preprocessing.image import img_to_array, load_img
max_iters = int(os.getenv('MAX_ITERS', '50000'))


def load_images_from_folder(folder, target_size=(28, 28)):
    images = []
    for filename in glob.glob(folder + '/*.jpg') + glob.glob(folder + '/*.png') + glob.glob(folder + '/*.jpeg'):
        img = load_img(filename, target_size=target_size, color_mode='grayscale')
        img = img_to_array(img)
        img = (img - 127.5) / 127.5  # Normalização
        images.append(img)
    return np.array(images)

#X and Z will have the real data and generated data respectlvely

X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 100])

def save_images(images, iter_num, directory='./images'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))

    plt.savefig(os.path.join(directory, f'batch_{iter_num}.png'))
    plt.close(fig)

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def generator(z):
    
    with tf.variable_scope("generator"):
        
        init = tf.contrib.layers.xavier_initializer()
        h1 = tf.layers.dense(inputs=z,units=128,activation=tf.nn.relu, kernel_initializer=init,use_bias=True)
        out = tf.layers.dense(inputs=h1,units=784,activation=tf.nn.tanh, kernel_initializer=init,use_bias=True)         

        return out

def discriminator(x):
    
    with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
        
        init = tf.contrib.layers.xavier_initializer()
        h1 = tf.layers.dense(inputs=x,units=128,activation=tf.nn.relu, kernel_initializer=init,use_bias=True)
        logits = tf.layers.dense(inputs=h1,units=1, kernel_initializer=init,use_bias=True)

        return logits
    
def sample_Z(r, c):
    return np.random.uniform(-1., 1., size=[r, c])

G_sample = generator(Z)
logits_real = discriminator(X)
logits_fake = discriminator(G_sample)
print(G_sample.shape, X.shape)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),
                                                                     logits=logits_real))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),
                                                                     logits=logits_fake))
D_loss = D_loss_real +D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake),
                                                                logits=logits_fake))

# Actual loss code for the equations but above is the better version 

D_real = tf.nn.sigmoid(logits_real)
D_fake = tf.nn.sigmoid(logits_fake)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

D_solver = tf.train.AdamOptimizer(learning_rate=1e-3,beta1=0.5).minimize(D_loss, var_list=D_vars)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-3,beta1=0.5).minimize(G_loss, var_list=G_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Carregar imagens
image_data = load_images_from_folder('./data/')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

print("Initial generated images")
samples = sess.run(G_sample,feed_dict={Z: sample_Z(128, 100)})
fig = show_images(samples[:16])
plt.show()
print()

for it in range(max_iters):
    
    if it % 1000 == 0:
        samples = sess.run(G_sample,feed_dict={Z: sample_Z(128, 100)})
        fig = save_images(samples[:16], it)
        plt.show()
        print()
    
    idx = np.random.randint(0, image_data.shape[0], 128)
    x = image_data[idx]
    x = x.reshape(-1, 784)
    #x, _ = mnist.train.next_batch(128)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: x, Z: sample_Z(128, 100)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(128, 100)})
    
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()

        