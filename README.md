---


# <img src="docs/images/CogKGE.png" alt="CogKGE: A Knowledge Graph Embedding Toolkit and Benckmark for Representing Multi-source and Heterogeneous Knowledge" width="20%"> 
**CogKGE: A Knowledge Graph Embedding Toolkit and Benckmark for Representing Multi-source and Heterogeneous Knowledge**

**Demo system and more information is available at http://cognlp.com/cogkge**


## Description   
CogKGE is a knowledge graph embedding toolkit that aims to represent **multi-source** and **heterogeneous** knowledge. CogKGE currently supports 17 models, 11 datasets including two multi-source heterogeneous KGs, five evaluation metrics, four knowledge adapters, four loss functions, three samplers and three built-in data containers.

This easy-to-use python package has the following advantages:

- **Multi-source and heterogeneous knowledge representation.** CogKGE explores the unified representation of knowledge from diverse sources. Moreover, our toolkit not only contains the triple fact-based embedding models, but also supports the fusion representation of additional information, including text descriptions, node types and temporal information.

- **Comprehensive models and benchmark datasets.** CogKGE implements lots of classic KGE models in the four categories of translation distance models, semantic matching models, graph neural network-based models and transformer-based models. Besides out-of-the-box models, we release two large benchmark datasets for further evaluating KGE methods, called EventKG240K and CogNet360K.
- **Extensible and modularized framework.** CogKGE provides a programming framework for KGE tasks. Based on the extensible architecture, CogKGE can meet the requirements of module extension and secondary development, and pre-trained knowledge embeddings can be directly applied to downstream tasks.
- **Open source and visualization demo.** Besides the toolkit, we also release an online system to discover knowledge visually. Source code, datasets and pre-trained embeddings are publicly available.

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
from cogktr import *

# loader lut
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

# loader model
model = BoxE(entity_dict_len=len(node_lut),
             relation_dict_len=len(relation_lut),
             embedding_dim=50)

# load predictor
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
