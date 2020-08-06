import time

from flask import Flask,Response,json,request
import tensorflow as tf
from BertModel.utils.BertWorker import BertWorker
import re
from bert_base.bert.tokenization import FullTokenizer
from extract_features import convert_lst_to_tokens

app = Flask(__name__)


from bert_base.server.helper import get_run_args
import os
import helpers


class BertMsg():
    def __init__(self,msg):
        self.msg = msg



#------------------------------------------------------------------------------------------

##获取模型运行参数
args = get_run_args()

num_labels, label2id, id2label = helpers.init_predict_var(args.classify_model_pb_dir)
classify_graph_path=os.path.join(args.classify_model_pb_dir,"classification_model.pb")
with tf.gfile.GFile(classify_graph_path, 'rb') as f:
    classify_graph_def = tf.GraphDef()
    classify_graph_def.ParseFromString(f.read())
classify_bertWorker = BertWorker(args, args.device_map, classify_graph_path, "CLASS", id2label,"","")


msg=[]
classify_bert = BertMsg(msg)
classify_estimator = classify_bertWorker.get_estimator(tf,classify_graph_def,"CLASS")
r = classify_estimator.predict(input_fn=classify_bertWorker.input_fn_builder(classify_bert),yield_single_examples=False)
msg=["classify_预热"]
classify_bert.msg = msg
classify_bertWorker.run_classify(r)


#------------------------------------------------------------------------------------------


num_labels, predicate_id2label, token_id2label = helpers.ner_init_predict_var(args.ner_model_pb_dir)
ner_graph_path=os.path.join(args.ner_model_pb_dir,"ner_model.pb")
with tf.gfile.GFile(ner_graph_path, 'rb') as f:
    ner_graph_def = tf.GraphDef()
    ner_graph_def.ParseFromString(f.read())

ner_bertWorker = BertWorker(args, args.device_map, ner_graph_def, "NER", "",predicate_id2label, token_id2label)
ner_estimator = ner_bertWorker.get_estimator(tf,ner_graph_def,"NER")

msg=["ner_预热数据"]
ner_bert = BertMsg(msg)

ner_r = ner_estimator.predict(input_fn=ner_bertWorker.ner_input_fn_builder(ner_bert),yield_single_examples=False)
ner_bertWorker.run_ner(ner_r)

schemas50FilePath="BertModel/checkpoints/all_50_schemas"
vocabfile="BertModel/checkpoints/vocab_ner.txt"
schemaDict = helpers.get50SchemasDict(schemas50FilePath)
tokenizer = FullTokenizer(vocabfile)

@app.route("/predict",methods=["POST"])
def predict():
    # text = request.form["text"]
    postForm = json.loads(request.get_data(as_text=True))
    text = postForm["text"]
    # text="北京市，简称“京”，古称燕京、北平，是中华人民共和国首都、"
    # text2 = "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部"

    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', '', text)
    sentences = text.strip().split("。")
    new_sentences = list(filter(helpers.filter_blank, sentences))
    classify_bert.msg = new_sentences
    if (len(new_sentences) != 0):
        ###关系抽取
        classify_reuslt = classify_bertWorker.run_classify(r)
        assert len(new_sentences) == len(classify_reuslt)
        if (len(classify_reuslt) != 0):
            # token_lable_input
            ner_input_texts = []
            sens = []
            for single_index, singletext_labels in enumerate(classify_reuslt):
                if (len(singletext_labels) != 0):
                    # 获取一句话中的关系
                    for label in singletext_labels:
                        sub_obj = schemaDict[label]
                        subject = sub_obj[0]
                        object = sub_obj[1]
                        tmp_sentence = subject + "，" + label + "，" + object + "。" + new_sentences[single_index] + "|||" + label
                        sens.append(new_sentences[single_index])
                        ner_input_texts.append(tmp_sentence)

            ner_bert.msg = ner_input_texts
            ####实体抽取#########
            ner_result = ner_bertWorker.run_ner(ner_r)
            ###########
            ner_tokens = convert_lst_to_tokens(ner_input_texts, 128, tokenizer)
            # print("tokens:",tokens)
            real_tokens_length = []
            for single_token in ner_tokens:
                for i, token in enumerate(single_token):
                    if (token == "[SEP]"):
                        real_tokens_length.append(i)
                        # print(real_tokens_length)
                        break

            result_list = []
            for sen_index, single_predicate_result in enumerate(ner_result):
                predicate_result = single_predicate_result[len(single_predicate_result) - 1][0]
                token_label = single_predicate_result[:len(single_predicate_result) - 1]
                # print(predicate_result)
                # print(token_label)

                # print(predicate_result)
                # print("token_label",token_label)
                sub_index_list = []
                obj_index_list = []

                sub_label_list = []
                obj_label_list = []

                for index, singe_token_label in enumerate(token_label):
                    if (singe_token_label == "[category]"):
                        break
                    words = singe_token_label.split("-")
                    if (len(words) == 2 and index < real_tokens_length[sen_index]):
                        if (words[1] == "SUB"):
                            sub_index_list.append(index)
                            sub_label_list.append(ner_tokens[sen_index][index + 1])
                        else:
                            obj_index_list.append(index)
                            obj_label_list.append(ner_tokens[sen_index][index + 1])

                # print(sub_index_list)
                # print(obj_index_list)
                sub_string = "".join(sub_label_list)
                obj_string = "".join(obj_label_list)
                if (sub_string != "" and obj_string != ""):
                    # resultFile.write("    text:::")
                    # resultFile.write(sens[sen_index])
                    # resultFile.write("\n")
                    # resultFile.write("     sub:::")
                    # resultFile.write("".join(sub_label_list))
                    # resultFile.write("\n")
                    # resultFile.write("relation:::")
                    # resultFile.write(predicate_result)
                    # resultFile.write("\n")
                    # resultFile.write("     obj:::")
                    # resultFile.write("".join(obj_label_list))
                    # resultFile.write("\n")
                    # resultFile.write("\n")
                    sub_type = schemaDict[predicate_result][0]
                    obj_type = schemaDict[predicate_result][1]
                    sub = "".join(sub_label_list)
                    relation = predicate_result
                    obj = "".join(obj_label_list)
                    tmp_result=[]
                    tmp_result.append(sub)
                    tmp_result.append(relation)
                    tmp_result.append(obj)
                    result_list.append(tmp_result)


    result={}
    result["result"] = result_list
    print(result)
    response = Response(json.dumps(result), content_type='application/json')

    return response


@app.route("/predict_ner",methods=["GET","POST"])
def predict_ner():

    text1="歌曲，作曲，人物。《离开》是由张宇谱曲，演唱|||作曲"
    # text2 = "影视作品,主演,人物。《娘家的故事第二部》是张玲执导，林在培、何赛飞等主演的电视剧|||主演"
    # text3="人物,身高,number。爱德华·尼科·埃尔南迪斯（1986-），是一位身高只有70公分哥伦比亚男子，体重10公斤，只比随身行李高一些，2010年获吉尼斯世界纪录正式认证，成为全球当今最矮的成年男人|||身高"
    # text4="人物，出生地，地点。查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部|||出生地"
    # text5="人物，丈夫，人物。李治即位后，萧淑妃受宠，王皇后为了排挤萧淑妃，答应李治让身在感业寺的武则天续起头发，重新纳入后宫|||丈夫"

    text=[text1]
    ner_bert.msg = text
    ner_bertWorker.run_ner(ner_r)
    return "aaa"



if __name__ == "__main__":
    app.run()
