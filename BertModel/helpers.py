#!/usr/bin/env python
# encoding: utf-8
# @Time    : 2020/7/20 13:41
# @Author  : lxx
# @File    : helpers.py
# @Software: PyCharm
import os
import logging
import sys
import zmq
from datetime import datetime
import pickle
from termcolor import colored

__all__ = ['__version__']
__version__ = '1.7.8'

def check_tf_version():
    import tensorflow as tf
    tf_ver = tf.__version__.split('.')
    assert int(tf_ver[0]) >= 1 and int(tf_ver[1]) >= 10, 'Tensorflow >=1.10 is required!'
    return tf_ver

_tf_ver_ = check_tf_version()

class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)

def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger



def import_tf(device_map=-1, verbose=False, use_fp16=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if len(device_map) < 0 else ",".join(device_map)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf

def init_predict_var(path):
    """
    初始化NER所需要的一些辅助数据
    :param path:
    :return:
    """
    label_list_file = os.path.join(path, 'label_list.pkl')
    label_list = []
    if os.path.exists(label_list_file):
        with open(label_list_file, 'rb') as fd:
            label_list = pickle.load(fd)
    num_labels = len(label_list)

    with open(os.path.join(path, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
    return num_labels, label2id, id2label





# class BertServer():
#     def __init__(self, args):
#         super().__init__()
#         self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)
#
#         self.max_seq_len = args.max_seq_len
#         self.num_worker = args.num_worker
#         self.max_batch_size = args.max_batch_size
#         self.num_concurrent_socket = max(8, args.num_worker * 2)  # optimize concurrency for multi-clients
#         self.port = args.port
#         self.args = args
#         self.status_args = {k: (v if k != 'pooling_strategy' else v.value) for k, v in sorted(vars(args).items())}
#         self.status_static = {
#             'tensorflow_version': _tf_ver_,
#             'python_version': sys.version,
#             'server_version': __version__,
#             'pyzmq_version': zmq.pyzmq_version(),
#             'zmq_version': zmq.zmq_version(),
#             'server_start_time': str(datetime.now()),
#         }
#         self.processes = []
#         # 如果BERT model path 不是空的，那么就启动bert模型
#         # if args.mode == 'BERT':
#         #     self.logger.info('freeze, optimize and export graph, could take a while...')
#         #     with Pool(processes=1) as pool:
#         #         # optimize the graph, must be done in another process
#         #         from .graph import optimize_bert_graph
#         #         self.graph_path = pool.apply(optimize_bert_graph, (self.args,))
#         #     # from .graph import optimize_graph
#         #     # self.graph_path = optimize_graph(self.args, self.logger)
#         #     if self.graph_path:
#         #         self.logger.info('optimized graph is stored at: %s' % self.graph_path)
#         #     else:
#         #         raise FileNotFoundError('graph optimization fails and returns empty result')
#         # elif args.mode == 'NER':
#         #     self.logger.info('lodding ner model, could take a while...')
#         #     with Pool(processes=1) as pool:
#         #         # optimize the graph, must be done in another process
#         #         from .graph import optimize_ner_model
#         #         num_labels, label2id, id2label = init_predict_var(self.args.model_dir)
#         #         self.num_labels = num_labels + 1
#         #         self.id2label = id2label
#         #         self.graph_path = pool.apply(optimize_ner_model, (self.args, self.num_labels))
#         #     if self.graph_path:
#         #         self.logger.info('optimized graph is stored at: %s' % self.graph_path)
#         #     else:
#         #         raise FileNotFoundError('graph optimization fails and returns empty result')
#         if args.mode == 'CLASS':
#             self.logger.info('lodding classification predict, could take a while...')
#
#             # optimize the graph, must be done in another process
#             # from .graph import optimize_class_model
#             num_labels, label2id, id2label = init_predict_var(self.args.model_dir)
#             self.num_labels = num_labels
#             self.id2label = id2label
#             self.logger.info('contain %d labels:%s' %(num_labels, str(id2label.values())))
#             self.graph_path = "D:\\LiuXianXian\\pycharm--code\\flask4bert\\BertModel\\model\\classification_model.pb"
#             if self.graph_path:
#                 self.logger.info('optimized graph is stored at: %s' % self.graph_path)
#             else:
#                 raise FileNotFoundError('graph optimization fails and returns empty result')
#         else:
#             raise ValueError('args model not special')