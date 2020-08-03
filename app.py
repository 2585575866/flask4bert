from flask import Flask
import tensorflow as tf
import tensorflow.keras as keras
import  numpy as np

from BertModel.Bert import BertWorker

app = Flask(__name__)


from bert_base.server.helper import get_run_args
import os
import pickle



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


#------------------------------------------------------------------------------------------
# args = get_run_args()
#
# # bertServer = BertServer(args)
# num_labels, label2id, id2label = init_predict_var(args.model_dir)
# classify_graph_path=os.path.join(args.model_pb_dir,"classification_model.pb")
# with tf.gfile.GFile(classify_graph_path, 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
# bertWorker = BertWorker(args, args.device_map, classify_graph_path, "CLASS", id2label,"","")
# estimator = bertWorker.get_estimator(tf,graph_def,"CLASS")
#
# class Bert():
#     def __init__(self,msg):
#         self.msg = msg
# msg=["预热数据"]
# bert = Bert(msg)
#
# r = estimator.predict(input_fn=bertWorker.input_fn_builder(bert),yield_single_examples=False)
# bertWorker.run(r)


#------------------------------------------------------------------------------------------



args = get_run_args()
num_labels, predicate_id2label, token_id2label = ner_init_predict_var(args.ner_model_pb_dir)
ner_graph_path=os.path.join(args.ner_model_pb_dir,"ner_model.pb")
with tf.gfile.GFile(ner_graph_path, 'rb') as f:
    ner_graph_def = tf.GraphDef()
    ner_graph_def.ParseFromString(f.read())

ner_bertWorker = BertWorker(args, args.device_map, ner_graph_def, "NER", "",predicate_id2label, token_id2label)
ner_estimator = ner_bertWorker.get_estimator(tf,ner_graph_def,"NER")

class Bert():
    def __init__(self,msg):
        self.msg = msg
msg=["ner_预热数据"]
ner_bert = Bert(msg)

ner_r = ner_estimator.predict(input_fn=ner_bertWorker.ner_input_fn_builder(ner_bert),yield_single_examples=False)
ner_bertWorker.run_ner(ner_r)







# @app.route("/predict",methods=["GET","POST"])
# def predict():
#
#     text1="北京市，简称“京”，古称燕京、北平，是中华人民共和国首都、"
#     text2 = "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部"
#     text=[text1,text2]
#     bert.msg = text
#     bertWorker.run(r)
#     return "aaa"


@app.route("/predict_ner",methods=["GET","POST"])
def predict_ner():

    text1="歌曲，作曲，人物。《离开》是由张宇谱曲，演唱|||作曲"
    text2 = "影视作品,主演,人物。《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧|||主演"
    text3="人物,身高,number。爱德华·尼科·埃尔南迪斯（1986-），是一位身高只有70公分哥伦比亚男子，体重10公斤，只比随身行李高一些，2010年获吉尼斯世界纪录正式认证，成为全球当今最矮的成年男人|||身高"
    text4="人物，出生地，地点。查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部|||出生地"
    text5="人物，丈夫，人物。李治即位后，萧淑妃受宠，王皇后为了排挤萧淑妃，答应李治让身在感业寺的武则天续起头发，重新纳入后宫|||丈夫"

    text=[text1,text2,text3,text4,text5]
    ner_bert.msg = text
    ner_bertWorker.run_ner(ner_r)
    return "aaa"



if __name__ == "__main__":
    app.run()
