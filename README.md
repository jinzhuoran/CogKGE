---

# <img src="docs/images/CogKGE.png" alt="CogKGE: A Knowledge Graph Embedding Toolkit and Benckmark for Representing Multi-source and Heterogeneous Knowledge" width="20%">

**CogKGE: A Knowledge Graph Embedding Toolkit and Benckmark for Representing Multi-source and Heterogeneous Knowledge**

**Demo system and more information is available at http://cognlp.com/cogkge**

## Description

CogKGE is a knowledge graph embedding toolkit that aims to represent **multi-source** and **heterogeneous** knowledge.
CogKGE currently supports 17 models, 11 datasets including two multi-source heterogeneous KGs, five evaluation metrics,
four knowledge adapters, four loss functions, three samplers and three built-in data containers.

This easy-to-use python package has the following advantages:

- **Multi-source and heterogeneous knowledge representation.** CogKGE explores the unified representation of knowledge
  from diverse sources. Moreover, our toolkit not only contains the triple fact-based embedding models, but also
  supports the fusion representation of additional information, including text descriptions, node types and temporal
  information.

- **Comprehensive models and benchmark datasets.** CogKGE implements lots of classic KGE models in the four categories
  of translation distance models, semantic matching models, graph neural network-based models and transformer-based
  models. Besides out-of-the-box models, we release two large benchmark datasets for further evaluating KGE methods,
  called EventKG240K and CogNet360K.
- **Extensible and modularized framework.** CogKGE provides a programming framework for KGE tasks. Based on the
  extensible architecture, CogKGE can meet the requirements of module extension and secondary development, and
  pre-trained knowledge embeddings can be directly applied to downstream tasks.
- **Open source and visualization demo.** Besides the toolkit, we also release an online system to discover knowledge
  visually. Source code, datasets and pre-trained embeddings are publicly available.

## Install

### Install from git

```bash
# clone CogKGE   
git clone https://github.com/jinzhuoran/CogKGE.git

# install CogKGE   
cd cogkge
pip install -e .   
pip install -r requirements.txt
```

### Install from pip

```bash
pip install cogkge
```

## Quick Start

### Pre-trained Embedder for Knowledge Discovery

```python
from cogkge import *

# loader lut
device = init_cogkge(device_id="0", seed=1)
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

# loader model
model = BoxE(entity_dict_len=len(node_lut),
             relation_dict_len=len(relation_lut),
             embedding_dim=50)

# load predictor
predictor = Predictor(model_name="BoxE",
                      data_name="EVENTKG2M",
                      model=model,
                      device=device,
                      node_lut=node_lut,
                      relation_lut=relation_lut,
                      pretrained_model_path="data/BoxE_Model.pkl",
                      processed_data_path="data",
                      reprocess=False,
                      fuzzy_query_top_k=10,
                      predict_top_k=10)

# fuzzy query node
result_node = predictor.fuzzy_query_node_keyword('champion')
print(result_node)

# fuzzy query relation
result_relation = predictor.fuzzy_query_relation_keyword("instance")
print(result_relation)

# query similary nodes
similar_node_list = predictor.predict_similar_node(node_id=0)
print(similar_node_list)

# given head and relation, query tail
tail_list = predictor.predcit_tail(head_id=0, relation_id=0)
print(tail_list)

# given tail and relation, query head
head_list = predictor.predict_head(tail_id=0, relation_id=0)
print(head_list)

# given head and tail, query relation
relation_list = predictor.predict_relation(head_id=0, tail_id=0)
print(relation_list)

# dimensionality reduction and visualization of nodes
visual_list = predictor.show_img(node_id=100, visual_num=1000)
```

### Programming Framework for Training Models

```python
import torch
from torch.utils.data import RandomSampler
from cogkge import *

device = init_cogkge(device_id="0", seed=1)

loader = EVENTKG2MLoader(dataset_path="../dataset", download=True)
train_data, valid_data, test_data = loader.load_all_data()
node_lut, relation_lut, time_lut = loader.load_all_lut()

processor = EVENTKG2MProcessor(node_lut, relation_lut, time_lut,
                               reprocess=True,
                               type=True, time=False, description=False, path=False,
                               time_unit="year",
                               pretrain_model_name="roberta-base", token_len=10,
                               path_len=10)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)
node_lut, relation_lut, time_lut = processor.process_lut()

train_sampler = RandomSampler(train_dataset)
valid_sampler = RandomSampler(valid_dataset)
test_sampler = RandomSampler(test_dataset)

model = TransE(entity_dict_len=len(node_lut),
               relation_dict_len=len(relation_lut),
               embedding_dim=50)

loss = MarginLoss(margin=1.0, C=0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

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

trainer = Trainer(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    train_sampler=train_sampler,
    valid_sampler=valid_sampler,
    model=model,
    loss=loss,
    optimizer=optimizer,
    negative_sampler=negative_sampler,
    device=device,
    output_path="../dataset",
    lookuptable_E=node_lut,
    lookuptable_R=relation_lut,
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
    load_checkpoint=None
)
trainer.train()

evaluator = Evaluatoraluator(
    test_dataset=test_dataset,
    test_sampler=test_sampler,
    model=model,
    device=device,
    metric=metric,
    output_path="../dataset",
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    lookuptable_E=node_lut,
    lookuptable_R=relation_lut,
    log=True,
    evaluator_batch_size=50000,
    dataloaderX=True,
    num_workers=4,
    pin_memory=True,
    trained_model_path=None
)
evaluator.evaluate()
```

## Model

<table class="greyGridTable" >
    <thead>
        <tr >
            <th >Category</th>
            <th >Model</th>
            <th>Conference</th>
            <th>Paper</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="7" >Translation Distance Models</td>
            <td>
                <a href="https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf">TransE</a> 
            </td>
            <td>NIPS 2013</td>
            <td>Translating embeddings for modeling multi-relational data</td>
        </tr>
        <tr>
            <td>
                <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.2800&rep=rep1&type=pdf">TransH</a> 
            </td>
            <td>AAAI 2014</td>
            <td>Knowledge Graph Embedding by Translating on Hyperplanes</td>
        </tr>
        <tr>
            <td>
                <a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523">TransR</a> 
            </td>
            <td>AAAI 2015</td>
            <td>Learning Entity and Relation Embeddings for Knowledge Graph Completion</td>
        </tr>
        <tr>
            <td>
                <a href="https://www.aclweb.org/anthology/P15-1067.pdf">TransD</a> 
            </td>
            <td>ACL 2015</td>
            <td>Knowledge Graph Embedding via Dynamic Mapping Matrix</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1509.05490.pdf">TransA</a> 
            </td>
            <td>AAAI 2015</td>
            <td>TransA: An Adaptive Approach for Knowledge Graph Embedding</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/2007.06267.pdf">BoxE</a> 
            </td>
            <td>NIPS 2020</td>
            <td>BoxE: A Box Embedding Model for Knowledge Base Completion</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/2011.03798.pdf">PairRE</a> 
            </td>
            <td>ACL 2021</td>
            <td>PairRE: Knowledge Graph Embeddings via Paired Relation Vectorss</td>
        </tr>
        <tr>
            <td rowspan="5">Semantic Matching Models</td>
            <td>
                <a href="https://icml.cc/2011/papers/438_icmlpaper.pdf">RESCAL</a> 
            </td>
            <td>ICML 2011</td>
            <td>A Three-Way Model for Collective Learning on Multi-Relational Data</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1412.6575.pdf">DistMult</a> 
            </td>
            <td> ICLR 2015</td>
            <td>Embedding Entities and Relations for Learning and Inference in Knowledge Bases</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1802.04868.pdf">SimpleIE</a> 
            </td>
            <td>NIPS 2018</td>
            <td>SimplE Embedding for Link Prediction in Knowledge Graphs</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1901.09590.pdf">TuckER</a> 
            </td>
            <td>ACL 2019</td>
            <td>TuckER: Tensor Factorization for Knowledge Graph Completion</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1902.10197.pdf">RotatE</a> 
            </td>
            <td>ICLR 2019</td>
            <td>RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space</td>
        </tr>
        <tr>
            <td rowspan="2">Graph Neural Network-based Models</td>
            <td>
                <a href="https://arxiv.org/pdf/1703.06103.pdf">R-GCN</a> 
            </td>
            <td>ESWC 2018</td>
            <td>Modeling Relational Data with Graph Convolutional Networks</td>
        </tr>
        <tr>
            <td>
                <a href="https://arxiv.org/pdf/1911.03082.pdf">CompGCN</a> 
            </td>
            <td>ICLR 2020</td>
            <td>Composition-based Multi-Relational Graph Convolutional Networks</td>
        </tr>
        <tr>
            <td rowspan="2">Transformer-based Models</td>
            <td>
                <a href="https://arxiv.org/pdf/2008.12813.pdf">HittER</a> 
            </td>
            <td>EMNLP 2021</td>
            <td>HittER: Hierarchical Transformers for Knowledge Graph Embeddings</td>
        </tr>
        <tr>
            <td>
                <a href="https://www.researchgate.net/profile/Jian-Tang-46/publication/337273572_KEPLER_A_Unified_Model_for_Knowledge_Embedding_and_Pre-trained_Language_Representation/links/6072896c299bf1c911c2051a/KEPLER-A-Unified-Model-for-Knowledge-Embedding-and-Pre-trained-Language-Representation.pdf">KEPLER</a> 
            </td>
            <td>TACL 2021</td>
            <td>KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation</td>
        </tr>
    </tbody>
</table>

## Dataset

### [EventKG240K](https://eventkg.l3s.uni-hannover.de/)

EventKG is a event-centric temporal knowledge graph, which incorporates over 690 thousand contemporary and historical
events and over 2.3 million temporal relations. To our best knowledge, EventKG240K is the first event-centric KGE
dataset. We use EventKG V3.0 data to construct the dataset. First, we filter entities and events based on their degrees.
Then, we select the triple facts when both nodes' degrees are greater than 10. At last, we add text descriptions and
node types for nodes and translate triples to quadruples by temporal information. The whole dataset contains 238,911
nodes, 822 relations and 2,333,986 triples.

### [CogNet360K](http://cognet.top/)

CogNet is a multi-source heterogeneous KG dedicated to integrating linguistic, world and commonsense knowledge. To build
a subset as the dataset, we count the number of occurrences for each node. Then, we sort frame instances by the minimum
occurrences of their connected nodes. After we have the sorted list, we filter the triple facts according to the preset
frame categories. Finally, we find the nodes that participate in these triple facts and complete their information. The
final dataset contains 360,637 nodes and 1,470,488 triples.

## Other KGE open-source project

- [Graphvite](https://graphvite.io/)
- [OpenKE](https://github.com/thunlp/OpenKE)
- [PyKEEN](https://github.com/SmartDataAnalytics/PyKEEN)
- [Pykg2vec](https://github.com/Sujit-O/pykg2vec)
- [LIBKGE](https://github.com/uma-pi1/kge)
- [KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)

