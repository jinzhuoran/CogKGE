from cogktr import *

device = init_cogktr(device_id="4", seed=1)

loader = EVENTKG2MLoader(dataset_path="../dataset", download=True)
train_data, valid_data, test_data = loader.load_all_data()
node_lut, relation_lut, time_lut = loader.load_all_lut()

processor = EVENTKG2MProcessor(node_lut, relation_lut, time_lut,
                               reprocess=True,
                               type=False, time=False, description=False, path=False,
                               time_unit="year",
                               pretrain_model_name="roberta-base", token_len=10,
                               path_len=10)
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
                          fuzzy_query_top_k=10,
                          predict_top_k=5)
###########################################################################
#                    以上为模型加载，以下为调用的接口
###########################################################################

# 模糊查询节点
result_node = predictor.fuzzy_query_node_keyword()
# result_node=predictor.fuzzy_query_node_keyword()
print(result_node)

# 模糊查询关系
result_relation = predictor.fuzzy_query_relation_keyword("sp")
# result_relation=predictor.fuzzy_query_relation_keyword()
print(result_relation)

# 查询相似节点
similar_node_list = predictor.predict_similar_node(node_id=0)
print(similar_node_list)

# 给出头节点和关系，查询尾节点
tail_list = predictor.predcit_tail(head_id=0, relation_id=0)
print(tail_list)
# 给出头结点，查询关系和尾节点
tail_list = predictor.predcit_tail(head_id=0)
print(tail_list)

# 给出头节点和关系，查询尾节点
head_list = predictor.predict_head(tail_id=0, relation_id=0)
print(head_list)
# 给出尾节点，查询关系和头节点
head_list = predictor.predict_head(tail_id=0)
print(head_list)

# 给出头节点和尾节点，查询关系
relation_list = predictor.predict_relation(head_id=0, tail_id=0)
print(relation_list)
