from pymongo import MongoClient

# internal package
import utils
import numpy as np
import tensorflow as tf

# database
client = MongoClient('localhost', 27017)
db = client.bheat
collection = db.origset
nm = 7
beats = list(collection.find({'class':6,
                              'bar': 128,
                              'gridicity': {'$lt': 0.4},
                              'diversity': {'$gt': 0.07}
                              }).limit(1002))

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
alll_f = alll[idxs]
# reshape to fit
alll_f = alll_f.reshape((alll_f.shape[0],alll_f.shape[1]*20,))
# okay
print "dataset: ", alll_f.shape

train_batch_size = 32
valid_batch_size = 32

# manage data loading
train_size = (alll_f.shape[0]/4)*3
val_size = (alll_f.shape[0]/4)

print "train/valid sets size: ", train_size, val_size, "\n"

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 100
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = alll_f.shape[1]

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with relu activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with relu activation #2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X


# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# Start Training
# Start a new TF session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = next_batch(batch_size, alll_f, np.array([-1] * alll_f.shape[0]))

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
            valid, _ = next_batch(1, alll_f, np.array([-1] * alll_f.shape[0]))
            g = sess.run(decoder_op, feed_dict={X: valid})

            utils.np_seq2mid(valid[0])
            utils.np_seq2mid(g[0])
