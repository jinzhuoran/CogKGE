import torch
from torch.utils.data import RandomSampler
from cogktr import *
import time

device=init_cogktr(device_id="3",seed=1)

loader =EVENTKG2MLoader(dataset_path="../dataset",download=True)
train_data, valid_data, test_data = loader.load_all_data()
node_lut, relation_lut ,time_lut= loader.load_all_lut()

processor = EVENTKG2MProcessor(node_lut, relation_lut,time_lut,
                               reprocess=False,
                               type=False,time=False,description=False,path=False,
                               time_unit="year",
                               pretrain_model_name="roberta-base",token_len=10,
                               path_len=10)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)
node_lut,relation_lut,time_lut=processor.process_lut()

model = BoxE(entity_dict_len=len(node_lut),
             relation_dict_len=len(relation_lut),
             embedding_dim=50)

loss = MarginLoss(margin=1.0,C=0)

metric = Link_Prediction(link_prediction_raw=True,
                         link_prediction_filt=False,
                         batch_size=5000000,
                         reverse=False)


predictor=Kr_Predictior(model=model,
                        pretrained_model_path="BoxE_Model.pkl",
                        device=device,
                        node_lut=node_lut,
                        relation_lut=relation_lut,
                        reprocess=False)
###########################################################################
#                    以上为模型加载，以下为调用的接口
###########################################################################
time_start = time.time()

#模糊查询节点
# result_node=predictor.fuzzy_query_node_keyword('Copa Colombia')
# result_node=predictor.fuzzy_query_node_keyword()
# print(result_node)

#模糊查询关系
# result_relation=predictor.fuzzy_query_relation_keyword("sport")
# result_relation=predictor.fuzzy_query_relation_keyword()
# print(result_relation)

#查询相似节点
# similar_node_list=predictor.predict_similar_node(node_id=0)
# print(similar_node_list)

#给出头节点和关系，查询尾节点
# tail_list=predictor.predcit_tail(head_id=0,relation_id=0)
# print(tail_list)
#给出头结点，查询关系和尾节点
tail_list=predictor.predcit_tail(head_id=0)
print(tail_list)

#给出头节点和关系，查询尾节点
# head_list=predictor.predict_head(tail_id=0,relation_id=0)
# print(head_list)
#给出尾节点，查询关系和头节点
head_list=predictor.predict_head(tail_id=0)
print(head_list)

#给出头节点和尾节点，查询关系
# relation_list=predictor.predict_relation(head_id=0,tail_id=0)
# print(relation_list)

time_end = time.time()
time_c = time_end - time_start  # 运行所花时间
print('time cost', time_c, 's')