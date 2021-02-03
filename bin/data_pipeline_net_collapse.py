"""Classes and functions to read, write and feed data."""

import os
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import csv
import json
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt

POSITIVE_EXAMPLES_PATH = 'positive'
NEGATIVE_EXAMPLES_PATH = 'negative'
# RECORD_REGEXP = re.compile(r'\d+\.tfrecords')


class DataWriter(object):

    """ Writes .tfrecords file to disk from window Stream objects.
    """

    def __init__(self, filename):
        self._writer = None
        self._filename = filename
        self._written = 0
        self._writer = tf.python_io.TFRecordWriter(self._filename)


    def write(self, sample_window, labels):
        try:
            n_samples = sample_window[:, 0].size
        except TypeError:
            n_samples = sample_window[0].data[0:400].size

        #n_traces = len(sample_window[0].data)
        try:
            n_traces = sample_window[0].size
        except AttributeError:
            n_traces = len(sample_window)

        # Extract data
        data = np.zeros((n_traces, n_samples), dtype=np.float32)
        #for i in range(n_traces):
        #    data[i, :] = sample_window[i].data[...]
        for i in range(n_traces):
            data[i] = sample_window[i].data[0:400]

        #used for one-hot label
        labels_traces = labels[:, 0].size
        label = np.zeros((labels_traces, n_samples), dtype=np.float32)
        for i in range(labels_traces):
            label[i] = labels[i]

        example = tf.train.Example(features=tf.train.Features(feature={
            'window_size': self._int64_feature(n_samples),
            'n_traces': self._int64_feature(n_traces),
            'data': self._bytes_feature(data.tobytes()),
            'label': self._bytes_feature(label.tobytes()),
        }))
        self._writer.write(example.SerializeToString())
        self._written += 1

    def close(self):
        self._writer.close()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(
                                    value=value.flatten().tolist()))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class DataReader(object):

    def __init__(self, path, config, shuffle=True):
        self._path = path
        self._shuffle = shuffle
        self._config = config
        self.win_size = config.win_size
        self.n_traces = config.n_traces
        self.labels_traces = config.labels_traces


        self._reader = tf.TFRecordReader()


    def read(self):
        filename_queue = self._filename_queue()
        _, serialized_example = self._reader.read(filename_queue)
        example = self._parse_example(serialized_example)
        return example

    def _filename_queue(self):
        fnames = []
        for root, dirs, files in os.walk(self._path):
            for f in files:
                if f.endswith(".tfrecords"):
                    fnames.append(os.path.join(root, f))
        fname_q = tf.train.string_input_producer(fnames,
                                                 shuffle=self._shuffle,
                                                 num_epochs=self._config.n_epochs)
#                                                  num_epochs=None)
        return fname_q

    def _parse_example(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'window_size': tf.FixedLenFeature([], tf.int64),
                'n_traces': tf.FixedLenFeature([], tf.int64),
                'data': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)})

        # Convert and reshape
        data = tf.decode_raw(features['data'], tf.float32)
        print(("data",data))
        data.set_shape([self.n_traces * self.win_size])
        data = tf.reshape(data, [self.n_traces, self.win_size])
        data = tf.transpose(data, [1, 0])
        # Pack
        features['data'] = data

        label = tf.decode_raw(features['label'], tf.float32)
        print("label.shape", label.shape)
        label.set_shape([self.labels_traces * self.win_size])
        print("self.labels_traces * self.win_size",self.labels_traces * self.win_size)
        #label = tf.reshape(label, [self.labels_traces, self.win_size])
        label = tf.reshape(label, [self.labels_traces, self.win_size])
        label = tf.transpose(label, [1, 0])
        print(("****data,label",data,label.shape))
        # Pack
        features['label'] = label
        return features

# x=tf.Print(samples,[samples,samples.shape,'input', samples],message='Debug message:',summarize=2)
class DataPipeline(object):

    """Creates a queue op to stream data for training.

    Attributes:
    samples: Tensor(float). batch of input samples [batch_size, n_channels, n_points]
    labels: Tensor(int32). Corresponding batch labels, [batch_size, n_channels, n_points]

    """

    def __init__(self, dataset_path, config, is_training):

        min_after_dequeue = 1000
        capacity = 1000 + 3 * config.batch_size

        if is_training:

            with tf.name_scope('inputs'):
                self._reader = DataReader(dataset_path, config=config)
                samples = self._reader.read()
                sample_input = samples['data']
                sample_target = samples["label"]
                self.samples, self.labels = tf.train.shuffle_batch(
                    [sample_input, sample_target],
                    batch_size=config.batch_size,
                    capacity=capacity,
                    min_after_dequeue=min_after_dequeue,
                    allow_smaller_final_batch=False)

        elif not is_training:

            with tf.name_scope('validation_inputs'):
                self._reader = DataReader(dataset_path, config=config)
                samples = self._reader.read()

                sample_input = samples["data"]
                sample_target = samples["label"]

                self.samples, self.labels= tf.train.batch(
                    [sample_input, sample_target],
                    batch_size=config.batch_size,
                    capacity=capacity,
                    num_threads=config.n_threads,
                    allow_smaller_final_batch=False)
        else:
            raise ValueError(
                "is_training flag is not defined, set True for training and False for testing")

