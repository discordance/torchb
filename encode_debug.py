from pymongo import MongoClient
from random import shuffle

# internal package
import utils
import numpy as np
import tensorflow as tf
import time

# database
client = MongoClient('localhost', 27017)
db = client.bheat
collection = db.origset

beats = list(collection.find({'class':6,
                              'bar': 128,
                              'density': {'$gt':0.01},
                              'diversity': {'$gt': 0.07},
                              'gridicity': {'$lt': 0.75}
                              }).limit(100))

# select random
# shuffle(beats)
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
alll_f = alll[idxs]
# alll_f = bincopy
# reshape to fit
# alll_f = alll_f.reshape((alll_f.shape[0],alll_f.shape[1]*20,))
alll_f = alll_f.reshape((alll_f.shape[0],alll_f.shape[1]*20,))
# okay
print "dataset: ", alll_f.shape

# manage data loading
train_size = (alll_f.shape[0]/4)*3
val_size = (alll_f.shape[0]/4)

print "train/valid sets size: ", train_size, val_size, "\n"

# will load the model
print "load the model"
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('encoded/jazz1/model_20000_0.003921.tb.meta')
    saver.restore(sess, tf.train.latest_checkpoint('encoded/jazz1/'))
    graph = tf.get_default_graph()
    is_training = graph.get_tensor_by_name("is_training:0")
    X = graph.get_tensor_by_name("input_layer:0")
    enc = graph.get_tensor_by_name("encoder_op:0")
    start_time = time.time()
    result = sess.run(enc, feed_dict={X:alll_f[:1],is_training:0})
    elapsed_time = time.time() - start_time
    print result,elapsed_time
