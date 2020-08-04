#!/usr/bin/env python
# encoding: utf-8
# @Time    : 2020/7/20 13:23
# @Author  : lxx
# @File    : Bert.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from termcolor import colored
import tensorflow as tf

import extract_features

from BertModel.utils.helpers import import_tf, set_logger



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

class BertWorker():
    def __init__(self, args, device_map, graph_path, mode, id2label,predicate_id2label, token_id2label):
        super().__init__()
        # self.worker_id = id
        self.device_map = device_map
        self.logger = set_logger(colored('WORKER-%d' , 'yellow'), args.verbose)
        self.max_seq_len = args.max_seq_len
        self.mask_cls_sep = args.mask_cls_sep
        self.daemon = True
        # self.exit_flag = multiprocessing.Event()
        # self.worker_address = worker_address_list
        # self.num_concurrent_socket = len(self.worker_address)
        # self.sink_address = sink_address
        self.prefetch_size = args.prefetch_size if len(self.device_map) > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.verbose = args.verbose
        self.graph_path = graph_path
        self.use_fp16 = args.fp16
        self.args = args
        self.mode = mode
        self.id2label = id2label
        self.predicate_id2label = predicate_id2label
        self.token_id2label = token_id2label

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def get_estimator(self, tf, graph_def,mode):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0'])

            return EstimatorSpec(mode=mode, predictions={
                'client_id': features['client_id'],
                'encodes': output[0]
            })

        def ner_model_fn(features, labels, mode, params):
            """
            命名实体识别模型的model_fn
            :param features:
            :param labels:
            :param mode:
            :param params:
            :return:
            """
            # with tf.gfile.GFile(self.graph_path, 'rb') as f:
            #     graph_def = tf.GraphDef()
            #     graph_def.ParseFromString(f.read())
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["input_type_ids"]
            print("++++++++++++++",input_ids)
            # tokens =features["tokens"]
            # print("++++++++++++++",tokens)
            input_map = {"input_ids": input_ids, "input_mask": input_mask,"segment_ids":segment_ids}
            predicate_probabilities, token_label_probabilities= tf.import_graph_def(graph_def, name='', input_map=input_map, return_elements=['predicate_probabilities:0','token_label_probabilities:0'])
            predicate_index = tf.argmax(predicate_probabilities, axis=-1, output_type=tf.int32)
            token_label_index = tf.argmax(token_label_probabilities, axis=-1)

            return EstimatorSpec(mode=mode, predictions={
                # 'client_id': features['client_id'],
                'predicate_index': predicate_index,
                'token_label_index': token_label_index,
                'predicate_probabilities': predicate_probabilities,
                'token_label_probabilities': token_label_probabilities,
                # "tokens":tokens

            })


        def classification_model_fn(features, labels, mode, params):
            """
            文本分类模型的model_fn
            :param features:
            :param labels:
            :param mode:
            :param params:
            :return:
            """
            # with tf.gfile.GFile(self.graph_path, 'rb') as f:
            #     graph_def = tf.GraphDef()
            #     graph_def.ParseFromString(f.read())
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            #为了兼容多输入，增加segment_id特征，即训练代码中的input_type_ids特征。
            # input_map = {"input_ids": input_ids, "input_mask": input_mask}
            segment_ids=features["input_type_ids"]
            input_map = {"input_ids": input_ids, "input_mask": input_mask,"segment_ids":segment_ids}
            pred_probs = tf.import_graph_def(graph_def, name='', input_map=input_map, return_elements=['pred_prob:0'])

            # return EstimatorSpec(mode=mode, predictions={
            #     'client_id': features['client_id'],
            #     'encodes': tf.argmax(pred_probs[0], axis=-1),
            #     'score': tf.reduce_max(pred_probs[0], axis=-1)
            # })

            return EstimatorSpec(mode=mode, predictions={
                # 'client_id': features['client_id'],
                'encodes': pred_probs[0],
                # 'score': tf.reduce_max(pred_probs[0], axis=-1)
            })

        # 0 表示只使用CPU 1 表示使用GPU
        config = tf.ConfigProto(device_count={'GPU': 0 if len(self.device_map) < 0 else len(self.device_map)})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        # session-wise XLA doesn't seem to work on tf 1.10
        # if args.xla:
        #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        if mode == 'NER':
            return Estimator(model_fn=ner_model_fn, config=RunConfig(session_config=config))
        elif mode == 'BERT':
            return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))
        elif mode == 'CLASS':
            return Estimator(model_fn=classification_model_fn, config=RunConfig(session_config=config))

    def run(self,r):
        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compatibility
        logger = set_logger(colored('WORKER-%d' , 'yellow'), self.verbose)
        logger.info('use device %s, load graph from %s' %
                    ('cpu' if len(self.device_map) <= 0 else ('gpu: %s' % ",".join(self.device_map)), self.graph_path))

        tf = import_tf(self.device_map, self.verbose, use_fp16=self.use_fp16)
        # estimator = self.get_estimator(tf)


        # for sock, addr in zip(receivers, self.worker_address):
        #     sock.connect(addr)

        # sink.connect(self.sink_address)

        predict_drop_remainder = False
        predict_file = "D:\\LiuXianXian\\pycharm--code\\flask4bert\\BertModel\\predictData\\predict.tf_record"
        # predict_input_fn = self.file_based_input_fn_builder(
        #     input_file=predict_file,
        #     seq_length=128,
        #     label_length=49,
        #     is_training=False,
        #     drop_remainder=predict_drop_remainder)
        # msg = "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部"
        # r = estimator.predict(input_fn=self.input_fn_builder(msg))

        prediction=next(r)
        pred_label_result = []
        pred_score_result = []
        for index, class_probabilitys in enumerate(prediction["encodes"]):
            single_result = []
            single_socre_result = []
            logger.info(prediction)
            pro_sum = 0.0
            for idx,class_probability in enumerate(class_probabilitys):
                pro_sum=pro_sum+class_probability
                if class_probability > 0.5:
                    single_result.append(self.id2label.get(idx, -1))
                    single_socre_result.append(class_probability)
            print(pro_sum)
            pred_label_result.append(single_result)
            pred_score_result.append(single_socre_result)
            # pred_label_result = [self.id2label.get(x, -1) for x in r['encodes'] ]

            # pred_score_result = r['score'].tolist()
            to_client = {'pred_label': pred_label_result, 'score': pred_score_result}
            logger.info(to_client)
            print("---break")

        # rst = send_ndarray(sink, r['client_id'], to_client)
        # logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))
        return "predict"

    def convert_id_to_label(self,pred_ids_result, idx2label, batch_size):
        """
        将id形式的结果转化为真实序列结果
        :param pred_ids_result:
        :param idx2label:
        :return:
        """
        result = []
        index_result = []
        for row in range(batch_size):
            curr_seq = []
            curr_idx = []
            ids = pred_ids_result[row]
            for idx, id in enumerate(ids):
                if id == 0:
                    break
                curr_label = idx2label[id]
                if curr_label in ['[CLS]', '[SEP]']:
                    if id == 102 and (idx < len(ids) and ids[idx + 1] == 0):
                        break
                    continue
                # elif curr_label == '[SEP]':
                #     break
                curr_seq.append(curr_label)
                curr_idx.append(id)
            result.append(curr_seq)
            index_result.append(curr_idx)
        return result, index_result

    def ner_result_to_json(self,predict_ids, id2label):
        """
        NER识别结果转化为真实标签结果进行返回
        :param predict_ids:
        :param id2label
        :return:
        """
        if False:
            return predict_ids
        pred_label_result, pred_ids_result = self.convert_id_to_label(predict_ids, id2label, len(predict_ids))
        return pred_label_result, pred_ids_result

    def run_ner(self,r):
        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compatibility
        logger = set_logger(colored('WORKER-%d' , 'yellow'), self.verbose)

        # logger.info('use device %s, load graph from %s' %
        #             ('cpu' if len(self.device_map) <= 0 else ('gpu: %s' % ",".join(self.device_map)), self.graph_path))

        tf = import_tf(self.device_map, self.verbose, use_fp16=self.use_fp16)

        prediction = next(r)


        logger.info(prediction["predicate_probabilities"])
        logger.info(prediction["predicate_probabilities"].shape)
        logger.info(prediction["predicate_index"])
        logger.info(prediction["token_label_probabilities"])
        logger.info(prediction["token_label_probabilities"].shape)
        logger.info(prediction["token_label_index"])
        # logger.info("tokens=========",r["tokens"])
        predicate_index = prediction["predicate_index"]
        token_label_index = prediction["token_label_index"]
        logger.info(self.predicate_id2label)

        predicate_result = []
        for tmp_predicate_index in predicate_index:
            tmp_result = []
            tmp_result.append(self.predicate_id2label.get(tmp_predicate_index, -1))
            predicate_result.append(tmp_result)
        logger.info(predicate_result)

        token_label_result, pred_ids_result = self.ner_result_to_json(token_label_index, self.token_id2label)
        logger.info(token_label_result)

        result = []
        for index, tmp_token_label_result in enumerate(token_label_result):
            # logger.info(predicate_result[index])
            tmp_token_label_result.append(predicate_result[index])
            result.append(tmp_token_label_result)

        result_dict={"pred_label":predicate_result,"token_label_result":token_label_result}
        result_dict={"pred_label":predicate_result}
        logger.info(result)

        # print('rst:', rst)
        # logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))


    def input_fn_builder(self,bert):
        import sys
        sys.path.append('..')

        from bert_base.bert.tokenization import FullTokenizer


        print(bert.msg)
        def gen():
            while True:
                # tokenizer = FullTokenizer(vocab_file=os.path.join(self.args.bert_model_dir, 'vocab.txt'))
                tokenizer = FullTokenizer(vocab_file="D:\\LiuXianXian\\pycharm--code\\flask4bert\\BertModel\\checkpoints\\vocab_classify.txt")
                # Windows does not support logger in MP environment, thus get a new logger
                # inside the process for better compatibility
                logger = set_logger(colored('WORKER-%d' , 'yellow'), self.verbose)



                # logger.info(msg)
                # logger.info("message===="+"  ".join(msg))
                # logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg), client_id))
                # check if msg is a list of list, if yes consider the input is already tokenized
                # 对接收到的字符进行切词，并且转化为id格式
                # logger.info('get msg:%s, type:%s' % (msg[0], type(msg[0])))
                # msg = "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部"
                is_tokenized = all(isinstance(el, list) for el in bert.msg)
                logger.info(is_tokenized)
                tmp_f = list(extract_features.convert_lst_to_features(bert.msg, self.max_seq_len, tokenizer, logger,
                                                                      is_tokenized, self.mask_cls_sep))
                print([f.input_ids for f in tmp_f])
                client_id ="1"
                yield {
                    # 'client_id': client_id,
                    'input_ids': [f.input_ids for f in tmp_f],
                    'input_mask': [f.input_mask for f in tmp_f],
                    'input_type_ids': [f.input_type_ids for f in tmp_f]
                }


        def input_fn():

            return (tf.data.Dataset.from_generator(
                gen,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'input_type_ids': tf.int32,
                              # 'client_id': tf.string
                               },
                output_shapes={
                    # 'client_id': (),
                    'input_ids': (None, self.max_seq_len),
                    'input_mask': (None, self.max_seq_len), #.shard(num_shards=4, index=4)
                    'input_type_ids': (None, self.max_seq_len)}).prefetch(self.prefetch_size))


        return input_fn





    def ner_input_fn_builder(self,ner_bert):
        import sys
        sys.path.append('..')

        from bert_base.bert.tokenization import FullTokenizer

        def gen():
            while True:
                tokenizer = FullTokenizer(vocab_file=os.path.join(self.args.bert_model_dir, 'vocab_ner.txt'))
                # Windows does not support logger in MP environment, thus get a new logger
                # inside the process for better compatibility
                logger = set_logger(colored('WORKER-%d' , 'yellow'), self.verbose)


                logger.info('ready and listening!')

                msg = ner_bert.msg
                # check if msg is a list of list, if yes consider the input is already tokenized
                # 对接收到的字符进行切词，并且转化为id格式
                # logger.info('get msg:%s, type:%s' % (msg[0], type(msg[0])))
                is_tokenized = all(isinstance(el, list) for el in msg)
                logger.info(is_tokenized)
                tmp_f = list(extract_features.ner_convert_lst_to_features(msg, self.max_seq_len, tokenizer, logger,
                                                                          is_tokenized, self.mask_cls_sep))


                print("tokens:",[f.tokens for f in tmp_f])
                print("input_ids:",[f.input_ids for f in tmp_f])
                print("--------------------------------")


                yield {
                    # 'client_id': client_id,
                    'input_ids': [f.input_ids for f in tmp_f],
                    'input_mask': [f.input_mask for f in tmp_f],
                    'input_type_ids': [f.input_type_ids for f in tmp_f],
                    # "tokens" : [f.tokens for f in tmp_f]
                }

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'input_type_ids': tf.int32,
                              # 'client_id': tf.string,
                              # "tokens":tf.string
                              },
                output_shapes={
                    # 'client_id': (),
                    'input_ids': (None, self.max_seq_len),
                    'input_mask': (None, self.max_seq_len), #.shard(num_shards=4, index=4)
                    'input_type_ids': (None, self.max_seq_len),
                   # "tokens":(None,self.max_seq_len)
                }).prefetch(self.prefetch_size))

        gen()
        print("aaa")
        return input_fn
