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


args = get_run_args()
# bertServer = BertServer(args)
num_labels, label2id, id2label = init_predict_var(args.model_dir)
classify_graph_path=os.path.join(args.model_pb_dir,"classification_model.pb")
with tf.gfile.GFile(classify_graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
bertWorker = BertWorker(args, args.device_map, classify_graph_path, "CLASS", id2label)
estimator = bertWorker.get_estimator(tf,graph_def)

class Bert():
    def __init__(self,msg):
        self.msg = msg
msg=["预热数据"]
bert = Bert(msg)

r = estimator.predict(input_fn=bertWorker.input_fn_builder(bert),yield_single_examples=False)
bertWorker.run(r)




















@app.route("/predict",methods=["GET","POST"])
def predict():

    text1="北京市，简称“京”，古称燕京、北平，是中华人民共和国首都、"
    text2 = "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部"
    text=[text1,text2]
    bert.msg = text
    bertWorker.run(r)
    return "aaa"



if __name__ == "__main__":
    app.run()
