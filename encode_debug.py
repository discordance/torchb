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
                              'gridicity': {'$lt': 0.5},
                              'diversity': {'$gt': 0.07}
                              }).limit(10))

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
