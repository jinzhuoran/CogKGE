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