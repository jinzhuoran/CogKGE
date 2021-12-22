import os
import torch
from torch.utils.data import RandomSampler
from cogktr import *

output_path = cal_output_path("../dataset/kr/FB15K/raw_data", "TransE")
if not os.path.exists(output_path):
    os.makedirs(output_path)
logger = save_logger(os.path.join(output_path, "run.log"))
logger.info("Data Path:{}".format("../dataset/kr/FB15K/raw_data"))
logger.info("Output Path:{}".format(output_path))
print("Data Path:{}".format("../dataset/kr/FB15K/raw_data"))
print("Output Path:{}".format(output_path))

device = str(7).strip().lower().replace('cuda:', '')
cpu = device == 'cpu'
if cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
elif device:  # non-cpu device requested
    os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
    assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
device = torch.device('cuda:0' if torch.cuda.is_available() == True else "cpu")


loader = FB15KLoader("../dataset/kr/FB15K/raw_data", True, "Research_code/CogKTR/dataset")
train_data, valid_data, test_data = loader.load_all_data()
node_vocab, relation_vocab = loader.load_all_vocabs()

processor = FB15K237Processor(node_vocab, relation_vocab)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)

train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

metric = Link_Prediction(link_prediction_raw= True,
                         link_prediction_filt=True,
                         batch_size=2000000,
                         reverse=False)

model = BoxE(entity_dict_len=len(node_vocab),
              relation_dict_len=len(relation_vocab),
               embedding_dim=50)

entity_candidate=list(node_vocab.word2idx.keys())
relation_candidate=list(relation_vocab.word2idx.keys())
test_entity=entity_candidate[6]
test_relation=relation_candidate[10]
print(test_entity)
print(test_relation)

evaluator = Kr_Evaluator(
    test_dataset=test_dataset,
    test_sampler=test_sampler,
    model=model,
    device=device,
    metric=metric,
    output_path=output_path,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    lookuptable_E= node_vocab,
    lookuptable_R= relation_vocab,
    logger=logger,
    evaluator_batch_size=50000,
    dataloaderX=True,
    num_workers=4,
    pin_memory=True,
    trained_model_path="/data/mentianyi/Research_code/CogKTR/dataset/kr/FB15K/experimental_output/BoxE2021-12-22--16-47-21.82--1000epochs/checkpoints/BoxE_100epochs",
)
evaluator.search_similar_entity(entity=test_entity,top=10)
evaluator.search_similar_head(tail=test_entity,relation=test_relation,top=10)
evaluator.search_similar_tail(head=test_entity,relation=test_relation,top=10)
evaluator.search_similar_head(tail=test_entity,top=10)
evaluator.search_similar_tail(head=test_entity,top=10)