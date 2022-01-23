import torch
from torch.utils.data import RandomSampler
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0]  # CogKGE root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add CogKGE root directory to PATH
from cogktr import *

device=init_cogktr(device_id="0",seed=1)

loader =EVENTKG2MLoader(dataset_path="../../dataset",download=True)
train_data, valid_data, test_data = loader.load_all_data()
node_lut, relation_lut ,time_lut= loader.load_all_lut()


processor = EVENTKG2MProcessor(node_lut, relation_lut,time_lut,
                               reprocess=True,
                               nodetype=False,time=False,description=False,graph=True,
                               time_unit="year",pretrain_model_name="roberta-base",token_len=10)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)
node_lut,relation_lut,time_lut=processor.process_lut()


train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

model = TransE(entity_dict_len=len(node_lut),
               relation_dict_len=len(relation_lut),
               embedding_dim=50)

loss = MarginLoss(margin=1.0,C=0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

metric = Link_Prediction(link_prediction_raw=True,
                         link_prediction_filt=False,
                         batch_size=5000000,
                         reverse=False)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, threshold_mode='abs', threshold=5,
    factor=0.5, min_lr=1e-9, verbose=True
)

negative_sampler = UnifNegativeSampler(triples=train_dataset,
                                       entity_dict_len=len(node_lut),
                                       relation_dict_len=len(relation_lut))

trainer = Kr_Trainer(
    train_dataset=train_dataset,
    valid_dataset=test_dataset,
    train_sampler=train_sampler,
    valid_sampler=test_sampler,
    model=model,
    loss=loss,
    optimizer=optimizer,
    negative_sampler=negative_sampler,
    device=device,
    output_path="../../dataset",
    lookuptable_E= node_lut,
    lookuptable_R= relation_lut,
    metric=metric,
    lr_scheduler=lr_scheduler,
    log=True,
    trainer_batch_size=100000,
    epoch=3000,
    visualization=1,
    apex=True,
    dataloaderX=True,
    num_workers=4,
    pin_memory=True,
    metric_step=200,
    save_step=200,
    metric_final_model=True,
    save_final_model=True,
    load_checkpoint= None
)
trainer.train()

