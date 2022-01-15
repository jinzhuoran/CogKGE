import torch
from torch.utils.data import RandomSampler
from cogktr import *

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


train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

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
#'Copa Colombia': [{'id': 79336, 'name': 'Copa Colombia', 'summary': 'The Copa Colombia (English: Colombia Cup; officially known as Copa BetPlay Dimayor is an annual football tournament in Colombia.'}],
result_node=predictor.fuzzy_query_node_keyword('Copa Colombia')
result_relation=predictor.fuzzy_query_relation_keyword("sport")
similar_node_list=predictor.predict_similar_node(node_id=0)
# print(similar_node_list)
tail_list=predictor.predcit_tail(head_id=0,relation_id=0)
# print(tail_list)
head_list=predictor.predict_head(tail_id=0,relation_id=0)
# print(head_list)
relation_list=predictor.predict_relation(head_id=0,tail_id=0)
# print(relation_list)








