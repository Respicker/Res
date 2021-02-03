# ============================================================== #
#                         Respicker eval                         #
#                                                                #
#                                                                #
# Eval resnet with processed dataset in seed format              #
#                                                                #
# Author Renyu Luo                                               #
# ============================================================== #


import data_pipeline_net as dp
from quakenet.data_io import load_stream
import numpy as np
from collections import Counter
import pandas as pd
import tensorflow as tf
import config as config
import argparse
import os, shutil
import time
from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
import tqdm
import glob
import setproctitle
import resnet
import fnmatch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from obspy.core import read



FLAGS = None
def load_datafiles(type):
    """
    Get all tfrecords from tfrecords dir:
    """

    tf_record_pattern = os.path.join(FLAGS.tfrecords_dir, '*.%s' % type)
    data_files = tf.gfile.Glob(tf_record_pattern)

    data_size = 0
    for fn in data_files:
        for record in tf.python_io.tf_record_iterator(fn):
            data_size += 1

    return data_files, data_size
def tree(top):
     #path,folder list,file list
    for path, names, fnames in os.walk(top):
        for fname in fnames:
            yield os.path.join(path, fname)
def fetch_window_data(stream,j):
    """fetch data from a stream window and dump in np array"""
    cfg = config.Config()
    data1 = np.empty((cfg.win_size+1, j))
    data = np.empty((cfg.win_size, j))
    for i in range(j):
        data1[:, i] = stream[i].data.astype(np.float32)
        #data[:, i] = stream[i][0:400].data
    for i in range(j):
        data[:, i] = data1[:,i][0:400]
    data = np.expand_dims(data, 0)
    return data
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result
def load_datafiles(type):
    """
    Get all tfrecords from tfrecords dir:
    """

    tf_record_pattern = os.path.join(FLAGS.tfrecords_dir, '*.%s' % type)
    data_files = tf.gfile.Glob(tf_record_pattern)

    data_size = 0
    for fn in data_files:
        for record in tf.python_io.tf_record_iterator(fn):
            data_size += 1

    return data_files, data_size
def preprocess_stream(stream):
    stream = stream.detrend('constant')
    stream = stream.normalize()
    stream = stream.filter('bandpass', freqmin=0.07, freqmax=20)
    return stream

def data_is_complete(stream):
    """Returns True if there is 1001*3 points in win"""
    cfg = config.Config()
    try:
        data_size = len(stream[0].data) + len(stream[1].data) + len(stream[2].data)
    except:
        data_size = 0
    data_lenth=int(cfg.win_size+1)*3
    if data_size == data_lenth:
        return True
    else:
        return False
def maybe_save_images(predict_images, images, filenames):
    """
    Save images to disk
    -------------
    Args:
        images: numpy array     [batch_size, image_size, image_size]
        filenames: numpy string array, filenames corresponding to the images   [batch_size]
    """

    if FLAGS.output_dir is not None:
        batch_size = predict_images.shape[0]
        for i in range(batch_size):
            image_array = predict_images[i, :]
            image_array1 = images[i, :, 1]
            print(image_array.shape, image_array1.shape)
            indexs = list(range(0, image_array.shape[0]))
            file_path = os.path.join(FLAGS.output_dir, filenames[i])
            ax = plt.subplot(211)
            plt.plot(indexs, image_array)
            plt.subplot(212, sharex=ax)
            plt.plot(indexs, image_array1)
            plt.savefig(file_path)
            plt.close()

def load_catalog(path):
    # type: (object) -> object
    """Loads a event catalog from a .csv file.

    Each row in the catalog references a know seismic event.

    Args:
        path: path to the input .csv file.

    Returns:
        catalog: A Pandas dataframe.
    """

    #catalog = pd.read_csv(path,sep = "\t",index_col=False)
    catalog = pd.read_csv(path)
    # Check if utc_timestamp exists, otherwise create it
    return catalog

def evaluate():
    """
    Eval resnet using specified args:
    """

    #data_files, data_size = load_datafiles(FLAGS.tfrecords_prefix)

    setproctitle.setproctitle('quakenet')

    tf.set_random_seed(1234)

    cfg = config.Config()
    cfg.add = 1
    cfg.n_clusters = FLAGS.num_classes
    cfg.n_clusters += 1
    model_files = [file for file in os.listdir(FLAGS.checkpoint_path) if
                   fnmatch.fnmatch(file, '*.meta')]
    for model_file in sorted(model_files):
        step = model_file.split(".meta")[0].split("-")[1]
        print (step)


    # stream data with a placeholder
    samples = {
        'data': tf.placeholder(tf.float32,
                                   shape=(cfg.batch_size, cfg.win_size, 3),
                                   name='input_data')}
    stream_path = FLAGS.stream_path
    try:
        #stream_files = [file for file in os.listdir(stream_path) if
        #                fnmatch.fnmatch(file, '*.mseed')]
        stream_files = [file for file in tree(stream_path) if
                        fnmatch.fnmatch(file, '*.seed')]
    except:
        stream_files = os.path.split(stream_path)[-1]
        print ("stream_files",stream_files)
    #data_files, data_size = load_datafiles(stream_path)
    n_events = 0
    time_start = time.time()
    print(" + Loading stream files {}".format(stream_files))

    events_dic = {"slice_start_time":[],
                  "P_pick": [],
                  "CP_pick": [],
                  "stname": [],
                  "utc_timestamp_p": [],
                  "utc_timestamp_s": [],
                  "utc_timestamp_cp": [],
                  "utc_timestamp_cs": [],
                  "S_pick": [],
                  "CS_pick": []
                  }

    with tf.Session() as sess:

        logits = resnet.resnet22130(samples['data'], FLAGS.batch_size, FLAGS.num_classes, False)
        time_start = time.time()
        catalog_name = "PS_pick_blocks.csv"
        output_catalog = os.path.join(FLAGS.output_dir, catalog_name)
        print('Catalog created to store events', output_catalog)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)


        saver = tf.train.Saver()
        model_file = os.path.join(FLAGS.checkpoint_path, model_file)
        if not tf.gfile.Exists(model_file):
            raise ValueError("Can't find checkpoint file")
        else:
            print('[INFO    ]\tFound checkpoint file, restoring model.')
            saver.restore(sess, model_file.split(".meta")[0])
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        predicted_images = resnet.predict(logits, cfg.batch_size, FLAGS.image_size)
        sess.graph.finalize()


        for stream_file in stream_files:
            print (" + Loading stream {}".format(stream_file))
            stream = read(stream_file)
            print (" -- Stream is ready, starting detection")
            try:

                i_str = 0

                while i_str < len(stream):
                    event_seed = stream[i_str:i_str + 3]

                    print("i:",i_str)

                    win_gen = event_seed.slide(window_length=FLAGS.window_size, step=FLAGS.window_step,
                                                  include_partial_windows=False)
                    for idx, win in enumerate(win_gen):
                        if data_is_complete(win):
                            #print (" + Preprocess stream")
                            win = preprocess_stream(win)
                            for i in range(len(win)):
                                if sum(win[i].data[0:400].flatten() == 0.0)>100:
                                    continue



                            to_fetch = [predicted_images, samples['data']]


                            # Feed window and fake cluster_id (needed by the net) but

                            fetch_data = np.zeros((16, 400, 3))
                            fetch_data_buffer = np.zeros((400, 3))
                            for i in range(3):
                                fetch_data_buffer[:,i] = win[i].data[0:400]

                            fetch_data[0] = fetch_data_buffer
                            feed_dict = {samples['data']: fetch_data}
                            predicted_images_value, images_value = sess.run(to_fetch, feed_dict)
                            clusters_p = np.where(predicted_images_value[0, :] == 1)
                            clusters_s = np.where(predicted_images_value[0, :] == 2)
                            clusters_cp = np.where(predicted_images_value[0, :] == 3)
                            clusters_cs = np.where(predicted_images_value[0, :] == 4)
                            clusters_n = np.where(predicted_images_value[0, :] == 0)

                            p_boxes = group_consecutives(clusters_p[0])
                            s_boxes = group_consecutives(clusters_s[0])
                            cp_boxes = group_consecutives(clusters_cp[0])
                            cs_boxes = group_consecutives(clusters_cs[0])
                            n_boxes = group_consecutives(clusters_n[0])
                            tp = []
                            ts = []
                            tcp = []
                            tcs = []
                            tpstamp = []
                            tsstamp = []
                            tcpstamp = []
                            tcsstamp = []



                            try:
                                if np.array(p_boxes).shape[1] > 50:
                                    for ip in range(len(p_boxes)):
                                        # print (len(p_boxes),p_boxes,p_boxes[ip])
                                        tpmean = float(min(p_boxes[ip]) / 200.00 + max(p_boxes[ip]) / 200.00)
                                        tp.append(tpmean)
                                        tpstamp = UTCDateTime(win[0].stats.starttime + tpmean).timestamp

                                if np.array(cp_boxes).shape[1] > 50:
                                    for ip in range(len(cp_boxes)):
                                        # print (len(p_boxes),p_boxes,p_boxes[ip])
                                        tcpmean = float(min(cp_boxes[ip]) / 200.00 + max(cp_boxes[ip]) / 200.00)
                                        tcp.append(tcpmean)
                                        tcpstamp = UTCDateTime(win[0].stats.starttime + tcpmean).timestamp

                                if np.array(s_boxes).shape[1] > 50:
                                    for iss in range(len(s_boxes)):
                                        tsmean = float(min(s_boxes[iss]) / 200.00 + max(s_boxes[iss]) / 200.00)
                                        ts.append(tsmean)
                                        tsstamp = UTCDateTime(win[0].stats.starttime + tsmean).timestamp
                                if np.array(cs_boxes).shape[1] > 50:
                                    for iss in range(len(cs_boxes)):
                                        tcsmean = float(min(cs_boxes[iss]) / 200.00 + max(cs_boxes[iss]) / 200.00)
                                        tcs.append(tcsmean)
                                        tcsstamp = UTCDateTime(win[0].stats.starttime + tcsmean).timestamp
                                #if np.array(p_boxes).shape[1] > 50 or np.array(cp_boxes).shape[1] > 50 or \
                                if np.array(cp_boxes).shape[1] > 50 or np.array(p_boxes).shape[1] > 50 or\
                                    np.array(s_boxes).shape[1] > 50 or np.array(cs_boxes).shape[1] > 50:
                                    events_dic["slice_start_time"].append(win[0].stats.starttime)
                                    events_dic["stname"].append(win[0].stats.station)
                                    events_dic["P_pick"].append(tp)
                                    events_dic["CP_pick"].append(tcp)
                                    events_dic["S_pick"].append(ts)
                                    events_dic["CS_pick"].append(ts)
                                    events_dic["utc_timestamp_p"].append(tpstamp)
                                    events_dic["utc_timestamp_s"].append(tsstamp)
                                    events_dic["utc_timestamp_cp"].append(tcpstamp)
                                    events_dic["utc_timestamp_cs"].append(tcsstamp)

                                    starttime = win[0].stats.starttime + 3600*8
                                    endtime = win[0].stats.endtime + 3600*8

                                    if UTCDateTime(cat['date'][a[n]] + cat['utc_time'][a[n]]) > starttime -10 and \
                                            UTCDateTime(cat['date'][a[n]] + cat['utc_time'][a[n]]) < endtime +10:

                                        win_filtered = win.copy()
                                        win_filtered2 = event_seed.slice(win[0].stats.starttime - 15,
                                                                         win[0].stats.endtime+25)


                                        lab = win_filtered2[2].copy()

                                        lab.stats.channel = "LAB"
                                        # lab =win[0].copy()

                                        print("predicted_images_value", predicted_images_value.shape)
                                        lab.data[...] = 0
                                        lab.data[...][1500:1900] = predicted_images_value[0, :]

                                        win_filtered2 = win_filtered2.normalize()
                                        win_filtered2 += lab
                                        if FLAGS.save_sac:
                                            output_seed = os.path.join(FLAGS.output_dir, "seed",
                                                                      "{}_{}.mseed".format(win_filtered2[0].stats.station,
                                                                                         str(win_filtered2[0].stats.starttime).replace(':',
                                                                                                                                      '_')))
                                            #win_filtered2.write(output_seed, format='MSEED')
                                            win_filtered2.plot(outfile=os.path.join(FLAGS.output_dir, "seed",
                                                                                    "{}_{}2.png".format(
                                                                                        win_filtered2[0].stats.station,
                                                                                        str(win_filtered2[
                                                                                                0].stats.starttime).replace(
                                                                                            ':',
                                                                                            '_'))))



                                        if FLAGS.plot:
                                            win_filtered.plot(outfile=os.path.join(FLAGS.output_dir, "viz",
                                                                                   "{}_{}.png".format(win_filtered[0].stats.station,
                                                                                                      str(win_filtered[
                                                                                                              0].stats.starttime).replace(':',
                                                                                                                  '_'))))
                                            win_filtered2.plot(outfile=os.path.join(FLAGS.output_dir, "viz",
                                                                                   "{}_{}2.png".format(
                                                                                       win_filtered[0].stats.station,
                                                                                       str(win_filtered[
                                                                                               0].stats.starttime).replace(
                                                                                           ':',
                                                                                           '_'))))


                                    # Wait for threads to finish.
                                    coord.join(threads)
                            except IndexError:
                                continue

                    i_str = i_str + 3

            except KeyboardInterrupt:
                print ('Interrupted at time {}.'.format(win[0].stats.starttime))
                print ("processed {} windows, found {} events".format(idx+1,n_events))
                print ("Run time: ", time.time() - time_start)
        df = pd.DataFrame.from_dict(events_dic)
        df.to_csv(output_catalog)

    print ("Run time: ", time.time() - time_start)
def main(_):
    """
    Run resnet prediction on input tfrecords
    """

    if FLAGS.output_dir is not None:
        if not tf.gfile.Exists(FLAGS.output_dir):
            print(
                '[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(FLAGS.output_dir)
    if FLAGS.plot:
        viz_dir=os.path.join(FLAGS.output_dir, "viz")
        if not tf.gfile.Exists(viz_dir):
            print(
                '[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(viz_dir)
    if FLAGS.plot:
        suc_dir=os.path.join(FLAGS.output_dir, "success")
        if not tf.gfile.Exists(suc_dir):
            print(
                '[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(suc_dir)
    if FLAGS.save_sac:
        sac_dir =os.path.join(FLAGS.output_dir,"seed")
        if not tf.gfile.Exists(sac_dir):
            print(
                '[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(sac_dir)

    if FLAGS.save_sac:
        plot_dir =os.path.join(FLAGS.output_dir, "plot")
        if not tf.gfile.Exists(plot_dir):
            print(
                '[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(plot_dir)
    evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval resnet on given tfrecords.')
    parser.add_argument('--stream_path', help='Tfrecords directory')
    parser.add_argument('--tfrecords_prefix', help='Tfrecords prefix', default='mseed')
    parser.add_argument('--checkpoint_path',
                        help='Path of checkpoint to restore. (Ex: ../Datasets/checkpoints/resnet.ckpt-80000)')
    parser.add_argument("--window_size",type=int,default=4,
                        help="size of the window to analyze")
    parser.add_argument("--window_step",type=int,default=2,
                        help="step between windows to analyze")
    parser.add_argument('--num_classes', help='Number of segmentation labels', type=int, default=6)
    parser.add_argument('--image_size', help='Target image size (resize)', type=int, default=400)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=16)
    parser.add_argument('--output_dir',
                        help='Output directory for the prediction files. If this is not set then predictions will not be saved')
    parser.add_argument("--plot", action="store_true",
                        help="pass flag to plot detected events in output")
    parser.add_argument("--save_sac", action="store_true",
                        help="pass flag to plot detected events in output")

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
