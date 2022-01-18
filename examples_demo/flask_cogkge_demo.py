# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
from flask import Flask, jsonify, request

app = Flask(__name__, static_url_path='/static')
app.config['JSON_AS_ASCII'] = False


# CORS(app, resources={r"/*": {"origins": "*"}}, send_wildcard=True, supports_credentials=True)

# 只接受POST方法访问


@app.route('/', methods=["GET"])
def index():
    return app.send_static_file('main.html')


# 模糊查询节点
@app.route('/fuzzy_query_node', methods=["GET", "POST"])
def fuzzy_query_node():
    if request.method == "POST":
        keyword = request.form['keyword']
    else:
        keyword = request.args['keyword']
    print(predictor.fuzzy_query_node_keyword(keyword))
    return jsonify(predictor.fuzzy_query_node_keyword(keyword))


# 模糊查询关系
@app.route('/fuzzy_query_relation', methods=["GET", "POST"])
def fuzzy_query_relation():
    if request.method == "POST":
        keyword = request.form['keyword']
    else:
        keyword = request.args['keyword']
    return jsonify(predictor.fuzzy_query_relation_keyword(keyword))


# 查询相似节点
@app.route('/predict_similar_node', methods=["GET", "POST"])
def predict_similar_node():
    if request.method == "POST":
        id = request.form['id']
    else:
        id = request.args['id']
    return jsonify(predictor.predict_similar_node(node_id=int(id)))


# 查询尾节点
@app.route('/predcit_tail', methods=["GET", "POST"])
def predcit_tail():
    if request.method == "POST":
        head_id = request.form['head_id']
        relation_id = request.form['relation_id']
    else:
        head_id = request.args['head_id']
        relation_id = request.args['relation_id']
    return jsonify(predictor.predcit_tail(head_id=int(head_id), relation_id=int(relation_id)))


# 查询关系
@app.route('/predict_relation', methods=["GET", "POST"])
def predcit_relation():
    if request.method == "POST":
        head_id = request.form['head_id']
        tail_id = request.form['tail_id']
    else:
        head_id = request.args['head_id']
        tail_id = request.args['tail_id']
    return jsonify(predictor.predict_relation(head_id=int(head_id), tail_id=int(tail_id)))


# 查询头节点
@app.route('/predict_head', methods=["GET", "POST"])
def predcit_head():
    if request.method == "POST":
        tail_id = request.form['tail_id']
        relation_id = request.form['relation_id']
    else:
        tail_id = request.args['tail_id']
        relation_id = request.args['relation_id']
    return jsonify(predictor.predict_head(tail_id=int(tail_id), relation_id=int(relation_id)))


if __name__ == "__main__":
    from cogktr import *

    device = init_cogktr(device_id="9", seed=1)

    loader = EVENTKG2MLoader(dataset_path="../dataset", download=True)
    train_data, valid_data, test_data = loader.load_all_data()
    node_lut, relation_lut, time_lut = loader.load_all_lut()

    processor = EVENTKG2MProcessor(node_lut, relation_lut, time_lut,
                                   reprocess=False,
                                   type=False, time=False, description=False, path=False,
                                   time_unit="year",
                                   pretrain_model_name="roberta-base", token_len=10,
                                   path_len=10)
    train_dataset = processor.process(train_data)
    valid_dataset = processor.process(valid_data)
    test_dataset = processor.process(test_data)
    node_lut, relation_lut, time_lut = processor.process_lut()

    model = BoxE(entity_dict_len=len(node_lut),
                 relation_dict_len=len(relation_lut),
                 embedding_dim=50)

    predictor = Kr_Predictior(model_name="BoxE",
                              data_name="EVENTKG2M",
                              model=model,
                              device=device,
                              node_lut=node_lut,
                              relation_lut=relation_lut,
                              pretrained_model_path="data/BoxE_Model.pkl",
                              processed_data_path="data",
                              reprocess=False,
                              fuzzy_query_top_k=20,
                              predict_top_k=40)
    app.run(host="0.0.0.0", port=5050)
