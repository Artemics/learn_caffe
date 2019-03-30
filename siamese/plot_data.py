'''
  Created on 23 Mar 2019 by Juntailang
'''
import matplotlib
matplotlib.use('Agg')
#To use it in the non-interactive backend; otherwise need python-tk.
#It seems that above import must precede that of caffe,
#as caffe implicitly imposes 'TkAgg' backend.

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import caffe
import lmdb
from PIL import Image

import numpy as np
import sys
import pdb


def set_model(path_proto, path_caffemodel):

  caffe.set_mode_cpu()
  net = caffe.Net(path_proto, path_caffemodel, caffe.TEST)

  return net


def read_lmdb(lmdb_env):

  cursor = lmdb_env.begin().cursor()
  
  datum = caffe.proto.caffe_pb2.Datum()

  for _, value in cursor:

    datum.ParseFromString(value)
    #pdb.set_trace()

    im = caffe.io.datum_to_array(datum)
    im_input = im[np.newaxis, np.newaxis, :, :]
    yield im_input, datum.label


def get_plotData(net, lmdb_dir):

  numRecords = lmdb_env.stat()['entries']
  dimFeat = 2
  result = np.zeros([numRecords,dimFeat])
  labels = np.full(numRecords, -1, dtype=np.int16)

  count = 0
  #pdb.set_trace()
  for im, label in read_lmdb(lmdb_dir):

    #if 0 == count%1000:
      #print(count)

    if np.random.rand()>0.1:
      continue

    net.blobs['data'].data[...] = im
    res = net.forward()
    # res is a dictionary object, whose only key is 'feat'
    # as far as I confirmed.
    result[count] = res['feat'][0]
    labels[count] = label
    count += 1

  return result, labels, count-1


if __name__ == '__main__':
  
  path_proto = sys.argv[1]
  path_caffemodel = sys.argv[2]
  lmdb_dir = sys.argv[3]

  #pdb.set_trace()
  net = set_model(path_proto, path_caffemodel)  

  lmdb_env = lmdb.open(lmdb_dir, readonly=True)

  featVectors, labels, last = get_plotData(net, lmdb_env)

#  for i in range(0,5):
#    print(featVectors[i], labels[i])

  featVectors_t = np.transpose(featVectors)
  fig, ax = plt.subplots()

  uniqueLabels = np.unique(labels)
  colors = cm.rainbow(np.linspace(0, 1, uniqueLabels.size)) 
   
  for label, c in zip(uniqueLabels, colors):
    if label == -1:
      continue
    idx = np.where(labels==label)
    #pdb.set_trace()
    ax.scatter(featVectors_t[0,idx], featVectors_t[1,idx]
      ,c=c, label=label)

  ax.legend()
  #plt.scatter(featVectors_t[0,:last],featVectors_t[1,:last],c=labels[0:last])
 
  plt.savefig('plot.png')


