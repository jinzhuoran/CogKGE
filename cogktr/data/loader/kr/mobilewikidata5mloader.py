import os
from ...datable import Datable
from ...lut import LookUpTable
from ....utils.download_utils import Download_Data
import json
from .baseloader import BaseLoader


class MOBILEWIKIDATA5MLoader(BaseLoader):
    def __init__(self, path, download=False, download_path=None):
        super().__init__(path, download, download_path,
                         train_name="mobilewikidata5m5000_transductive_train.txt",
                         valid_name="mobilewikidata5m5000_transductive_valid.txt",
                         test_name="mobilewikidata5m5000_transductive_test.txt")
        self.text_name = "mobilewikidata5m5000_text"

    def download_action(self):
        self.downloader.MOBILEWIKIDATA5M()

    def _load_lut(self,path):
        total_path = os.path.join(self.path, path).replace('\\', '/')
        lut = LookUpTable()
        lut.read_csv(total_path,sep='\t',names=["descriptions"],index_col=0)
        return lut

    def load_node_lut(self):
        pre_node_lut = os.path.join(self.path,"node_lut.pkl")
        if os.path.exists(pre_node_lut):
            node_lut = LookUpTable()
            node_lut.read_from_pickle(pre_node_lut)
        else:
            node_lut = self._load_lut(self.text_name)
            node_lut.add_vocab(self.node_vocab)
            node_lut.save_to_pickle(pre_node_lut)
        return node_lut

    def load_all_lut(self):
        """
        remember that in wikidata,we only have node lut
        :return: node lut (!!There is no relation lut!!)
        """
        node_lut = self.load_node_lut()
        relation_lut = LookUpTable()
        relation_lut.add_vocab(self.relation_vocab)
        return node_lut,relation_lut




# class MOBILEWIKIDATA5MLoader:
#     def __init__(self, path,download=False,download_path=None):
#         self.path = path
#         self.download = download
#         self.download_path=download_path
#         self.entity_list = list()
#         self.relation_list = list()
#         if self.download == True:
#             downloader = Download_Data(dataset_path=self.download_path)
#             downloader.MOBILEWIKIDATA5M()
#         self.train_name="mobilewikidata5m5000_transductive_train.txt"
#         self.valid_name="mobilewikidata5m5000_transductive_valid.txt"
#         self.test_name="mobilewikidata5m5000_transductive_test.txt"
#         self.text_name="mobilewikidata5m5000_text"
#
#
#     def _load_data(self, path):
#         heads = []
#         relations = []
#         tails = []
#         total_path = os.path.join(self.path, path).replace('\\', '/')
#         with open(total_path) as file:
#             for line in file:
#                 h, r, t = line.strip().split("\t")
#                 heads.append(h)
#                 relations.append(r)
#                 tails.append(t)
#                 self.entity_list.append(h)
#                 self.entity_list.append(t)
#                 self.relation_list.append(r)
#         datable = Datable()
#         datable(["head", "relation", "tail"], [heads, relations, tails])
#         return datable
#
#     def load_train_data(self):
#         train_data = self._load_data(self.train_name)
#         return train_data
#
#     def load_valid_data(self):
#         valid_data = self._load_data(self.valid_name)
#         return valid_data
#
#     def load_test_data(self):
#         test_data = self._load_data(self.test_name)
#         return test_data
#
#     def load_all_data(self):
#         train_data = self._load_data(self.train_name)
#         valid_data = self._load_data(self.valid_name)
#         test_data = self._load_data(self.test_name)
#         return train_data, valid_data, test_data
#
#     def _load_text(self):
#         entity_text_dict=dict()
#         total_path = os.path.join(self.path, self.text_name).replace('\\', '/')
#         with open(total_path) as file:
#             for line in file:
#                 entity_text_list= line.strip().split("\t")
#                 entity=entity_text_list[0]
#                 text=[str(i)+"\t" for i in entity_text_list]
#                 text="\t".join(text[1:])
#                 entity_text_dict[entity]=text
#         return entity_text_dict
#
#
#     def _load_lut(self, path, category=None):
#         entity_text_dict=list()
#         total_path = os.path.join(self.path, path).replace('\\', '/')
#         if not os.path.exists(total_path):
#             if category == "entity":
#                 print("Creating entities.json...")
#                 entity_name_list = list(set(list(self.entity_list)))
#                 # entity_name_list.sort(key=list(self.entity_list).index)
#                 lookuptable = LookUpTable()
#                 lookuptable.create_table(create_dic=True, item_list=entity_name_list)
#                 entities_dict = dict()
#                 for i in range(len(lookuptable)):
#                     entities_dict[lookuptable["name"][i]] = i
#                 json.dump(entities_dict, open(total_path, "w"), indent=4, sort_keys=True)
#
#
#
#             if category == "relation":
#                 print("Creating relations.json...")
#                 relation_name_list = list(set(list(self.relation_list)))
#                 # relation_name_list.sort(key=list(self.relation_list).index)
#                 lookuptable = LookUpTable()
#                 lookuptable.create_table(create_dic=True, item_list=relation_name_list)
#                 relations_dict = dict()
#                 for i in range(len(lookuptable)):
#                     relations_dict[lookuptable["name"][i]] = i
#                 json.dump(relations_dict, open(total_path, "w"), indent=4, sort_keys=True)
#
#         if category == "entity":
#             with open(total_path) as file:
#                 entity2idx = json.load(file)
#             print("Loading entity_descriptions...")
#             entity_text_dict=self._load_text()
#             lookuptable = LookUpTable()
#             lookuptable.create_table(create_dic=False, str_dic=entity2idx)
#             lookuptable("descriptions",entity_text_dict)
#             return lookuptable
#         if category == "relation":
#             with open(total_path) as file:
#                 relation2idx = json.load(file)
#             lookuptable = LookUpTable()
#             lookuptable.create_table(create_dic=False, str_dic=relation2idx)
#             return lookuptable
#
#     def load_entity_lut(self):
#         entity2idx = self._load_lut(path="entities.json", category="entity")
#         return entity2idx
#
#     def load_relation_lut(self):
#         relation2idx = self._load_lut(path="relations.json", category="realtion")
#         return relation2idx
#
#     def load_all_lut(self):
#         entity2idx = self._load_lut(path="entities.json", category="entity")
#         relation2idx = self._load_lut(path="relations.json", category="relation")
#         return entity2idx, relation2idx
