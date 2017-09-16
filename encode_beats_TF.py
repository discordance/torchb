from pymongo import MongoClient
from random import shuffle

# internal package
import utils
import numpy as np
import tensorflow as tf

# database
client = MongoClient('localhost', 27017)
db = client.bheat
collection = db.origset

beats = list(collection.find({'class':6,
                              'bar': 128,
                              'density': {'$gt':0.02},
                              'diversity': {'$gt': 0.07},
                              'gridicity': {'$lt': 0.5}
                              }))

# select random
shuffle(beats)
# beats = beats[:20]

# decompress in numpy
alll = []
for i, beat in enumerate(beats):
    np_beat = utils.decompress(beat['zip'], beat['bar'])
    for j, np_bar in enumerate(np_beat):
        alll.append(np_bar)
alll = np.array(alll)
print alll.shape

# get only the binary perc part of the beat
bincopy = alll[:,:,:15]
print bincopy.shape

# get only uniques
uniques, idxs = np.unique(bincopy, axis=0, return_index=True)
# alll_f = alll[idxs]
alll_f = bincopy
# reshape to fit
# alll_f = alll_f.reshape((alll_f.shape[0],alll_f.shape[1]*20,))
alll_f = alll_f.reshape((alll_f.shape[0],alll_f.shape[1]*15,))
# okay
print "dataset: ", alll_f.shape

# manage data loading
train_size = (alll_f.shape[0]/4)*3
val_size = (alll_f.shape[0]/4)

print "train/valid sets size: ", train_size, val_size, "\n"

# Training Parameters
learning_rate = 0.0001
num_steps = 200000
batch_size = 64

display_step = 2000

# Network Parameters
num_hidden_1 = 768 # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features
num_input = alll_f.shape[1]

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])
is_training = tf.placeholder(tf.bool, name='is_training')

weights = {
    'encoder_h1': tf.get_variable("encoder_h1", shape=[num_input, num_hidden_1], initializer=tf.contrib.layers.xavier_initializer()),
    'encoder_h2': tf.get_variable("encoder_h2", shape=[num_hidden_1, num_hidden_2], initializer=tf.contrib.layers.xavier_initializer()),
    'decoder_h1': tf.get_variable("decoder_h1", shape=[num_hidden_2, num_hidden_1], initializer=tf.contrib.layers.xavier_initializer()),
    'decoder_h2': tf.get_variable("decoder_h2", shape=[num_hidden_1, num_input], initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'encoder_b1': tf.get_variable("encoder_b1", shape=[num_hidden_1], initializer=tf.contrib.layers.xavier_initializer()),
    'encoder_b2': tf.get_variable("encoder_b2", shape=[num_hidden_2], initializer=tf.contrib.layers.xavier_initializer()),
    'decoder_b1': tf.get_variable("decoder_b1", shape=[num_hidden_1], initializer=tf.contrib.layers.xavier_initializer()),
    'decoder_b2': tf.get_variable("decoder_b2", shape=[num_input], initializer=tf.contrib.layers.xavier_initializer()),
}

# Building the encoder
def encoder(x):
    # Encoder Input layer
    l1_op = tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1'])
    layer_1 = tf.nn.relu(l1_op)

    # Encoder Hidden layer
    l2_op = tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2'])
    layer_2 = tf.nn.relu(l2_op)
    layer_2 = tf.layers.dropout(layer_2, 0.5)

    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer
    l1_op = tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1'])
    layer_1 = tf.nn.relu(l1_op)
    layer_1 = tf.layers.dropout(layer_1, 0.5)

    # Decoder Out layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2'])
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

#check accuracy
correct_prediction = tf.equal(tf.argmax(y_true,1), tf.argmax(y_pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define loss and optimizer
def loss_func():
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(cross_ent)

loss = loss_func()
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]

    return np.asarray(data_shuffle)

# saver
saver = tf.train.Saver()

# Start Training
# Start a new TF session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x = next_batch(batch_size, alll_f[:train_size])
        val_x = next_batch(batch_size, alll_f[train_size:])

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, is_training: 1})
        vl = sess.run(loss, feed_dict={X: val_x, is_training: 0})
        acc = sess.run(accuracy, feed_dict={X: val_x, is_training: 0})

        if i % display_step == 0:
            saver.save(sess, "encoded/jazz1/model_%i_%f.tb"% (i, l))

        if i % (display_step/32) == 0:
            print('Step %i: Minibatch Loss: %f, Acc: %f, Valid Loss: %f' % (i, l, acc, vl))

        if i % (display_step) == 0:
            g = sess.run(decoder_op, feed_dict={X: val_x, is_training: 0})
            print "ORIG: \n"
            print utils.draw(val_x[0].reshape((128,15)))
            print "REBUILD: \n"
            print utils.draw(g[0].reshape((128,15))) + "\n"
