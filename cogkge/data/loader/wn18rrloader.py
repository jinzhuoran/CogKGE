from .baseloader import BaseLoader


class WN18RRLoader(BaseLoader):
    def __init__(self, path, download=False, download_path=None):
        super().__init__(path, download, download_path,
                         train_name="train.txt",
                         valid_name="valid.txt",
                         test_name="test.txt")

    def download_action(self):
        self.downloader.WN18RR()

# import os
# from ...datable import Datable
# from ...lut import LookUpTable
# from ....utils.download_utils import Download_Data
# import json
#
#
# class WN18RRLoader:
#     def __init__(self, path,download=False,download_path=None):
#         self.path = path
#         self.download = download
#         self.download_path=download_path
#         self.entity_list = list()
#         self.relation_list = list()
#         if self.download == True:
#             downloader = Download_Data(dataset_path=self.download_path)
#             downloader.WN18RR()
#         self.train_name="train.txt"
#         self.valid_name="valid.txt"
#         self.test_name="test.txt"
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
#     def _load_lut(self, path, category=None):
#         total_path = os.path.join(self.path, path).replace('\\', '/')
#         if not os.path.exists(total_path):
#             if category == "entity":
#                 print("Creating entities.json...")
#                 entity_name_list = list(set(list(self.entity_list)))
#                 entity_name_list.sort(key=list(self.entity_list).index)
#                 lookuptable = LookUpTable()
#                 lookuptable.create_table(create_dic=True, item_list=entity_name_list)
#                 entities_dict = dict()
#                 for i in range(len(lookuptable)):
#                     entities_dict[lookuptable["name"][i]] = i
#                 json.dump(entities_dict, open(total_path, "w"), indent=4, sort_keys=True)
#
#             if category == "relation":
#                 print("Creating relations.json...")
#                 relation_name_list = list(set(list(self.relation_list)))
#                 relation_name_list.sort(key=list(self.relation_list).index)
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
#             lookuptable = LookUpTable()
#             lookuptable.create_table(create_dic=False, str_dic=entity2idx)
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
