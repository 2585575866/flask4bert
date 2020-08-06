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

from flask import json
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
    初始化所需要的一些辅助数据
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


def ner_init_predict_var(path):
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

    with open(os.path.join(path, 'predicate_label2id.pkl'), 'rb') as rf:
        predicate_label2id = pickle.load(rf)
        predicate_id2label = {value: key for key, value in predicate_label2id.items()}
    with open(os.path.join(path, 'token_label2id.pkl'), 'rb') as rf:
        token_label2id = pickle.load(rf)
        token_id2label = {value: key for key, value in token_label2id.items()}
    return num_labels, predicate_id2label,token_id2label


def filter_blank(x):
    return x !="" and x!=" "


def get50SchemasDict(schemaFilePath):
    dict={}
    for line in  open(schemaFilePath,encoding="utf-8"):
        tmpDict = json.loads(line)
        dict[tmpDict["predicate"]]=[tmpDict["subject_type"],tmpDict["object_type"]]
    return dict