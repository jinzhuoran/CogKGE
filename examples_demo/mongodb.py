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
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)
node_lut, relation_lut, time_lut = processor.process_lut()

model = BoxE(entity_dict_len=len(node_lut),
             relation_dict_len=len(relation_lut),
             embedding_dim=50)

loss = MarginLoss(margin=1.0, C=0)

metric = Link_Prediction(link_prediction_raw=True,
                         link_prediction_filt=False,
                         batch_size=5000000,
                         reverse=False)

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
# for value in predictor.detailed_relation_dict.values():
#     value['idd'] = str(value['id'])
#     del value['id']
#     value['name'] = str(value['name'])
#     value['summary'] = str(value['summary'])
#     predictor.insert_relation(value)

predictor.fuzzy_query_node_keyword('tom')
predictor.fuzzy_query_relation_keyword('city')

print(1)
