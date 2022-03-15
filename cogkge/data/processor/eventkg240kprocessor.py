import os
import pickle

import torch
from tqdm import tqdm
from transformers import RobertaModel
from transformers import RobertaTokenizer

from .baseprocessor import BaseProcessor
from ..dataset import Cog_Dataset
from ..lut import LookUpTable
from ..vocabulary import Vocabulary


class EVENTKG240KProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut, time_lut,mode,
                 reprocess=True,
                 # time=False, nodetype=False, description=False,,relationtype=False,
                 graph=False,
                 time_unit="year",
                 pretrain_model_name="roberta-base", token_len=10):
        """
        :param vocabs: node_vocab,relation_vocab,time_vocab
        """
        if mode not in ["type","description","time","normal"]:
            raise ValueError("{} mode is not supported!".format(mode))
        node_dict = {"type": False, "description": False, "time": False, "normal": False, mode: True}
        super().__init__("EVENTKG240K", node_lut, relation_lut, reprocess,
                         time=node_dict["time"], nodetype=node_dict["type"], description=node_dict["description"], graph=graph)
        self.time_lut = time_lut
        self.time_vocab = time_lut.vocab
        self.relationtype = node_dict["type"]

        self.time_unit = time_unit
        self.pre_training_model_name = pretrain_model_name
        self.token_length = token_len
        self.tokenizer = RobertaTokenizer.from_pretrained(self.pre_training_model_name)
        self.pre_training_model = RobertaModel.from_pretrained(self.pre_training_model_name)

        self.node_type_vocab = Vocabulary()
        self.relation_type_vocab = Vocabulary()
        self.node_type_vocab.buildVocab(list(self.node_lut.data['type']))
        self.relation_type_vocab.buildVocab(list(self.relation_lut.data["name"]))

        preprocessed_node_lut_file = os.path.join(self.processed_path, "processed_node_lut.pkl")
        preprocessed_relation_lut_file = os.path.join(self.processed_path, "processed_relation_lut.pkl")

        if not self.reprocess and os.path.exists(preprocessed_node_lut_file) and os.path.exists(
                preprocessed_relation_lut_file):
            self.node_lut = LookUpTable()
            self.node_lut.read_from_pickle(preprocessed_node_lut_file)
            self.relation_lut = LookUpTable()
            self.relation_lut.read_from_pickle(preprocessed_relation_lut_file)

        if self.description:
            if self.reprocess or not os.path.exists(preprocessed_node_lut_file):
                tokens_list = []
                masks_list = []
                for i in tqdm(range(len(self.node_lut))):
                    encoded_text = self.tokenizer.encode_plus(
                        str(self.node_lut.data["description"][i]),
                        add_special_tokens=True,
                        max_length=self.token_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    tokens_list.append(encoded_text["input_ids"])
                    masks_list.append(encoded_text['attention_mask'])
                self.node_lut.add_column(tokens_list, "input_ids")
                self.node_lut.add_column(masks_list, "attention_mask")
                self.node_lut.add_token(torch.cat(tokens_list, dim=0))
                self.node_lut.add_mask(torch.cat(masks_list, dim=0))
                self.node_lut.save_to_pickle(preprocessed_node_lut_file)

        if self.nodetype:
            if self.reprocess or not os.path.exists(preprocessed_node_lut_file):
                node_type_list = []
                for i in tqdm(range(len(node_lut))):
                    label = self.node_lut["type"][i]
                    label_id = torch.unsqueeze(torch.tensor(self.node_type_vocab.word2idx[label]), dim=0)
                    node_type_list.append(label_id)
                self.node_lut.add_type(torch.cat(node_type_list, dim=0))
                self.node_lut.save_to_pickle(preprocessed_node_lut_file)

            if self.reprocess or not os.path.exists(preprocessed_relation_lut_file):
                relation_type_list = []
                for i in tqdm(range(len(self.relation_lut))):
                    label = self.relation_lut["name"][i]
                    label_id = torch.unsqueeze(torch.tensor(self.relation_type_vocab.word2idx[label]), dim=0)
                    relation_type_list.append(label_id)
                self.relation_lut.add_type(torch.cat(relation_type_list, dim=0))
                self.relation_lut.save_to_pickle(preprocessed_relation_lut_file)

        if self.time:
            if time_unit == "day":
                pass
            if time_unit == "month":
                time_list = []
                month_time_vocab = Vocabulary()
                for key in tqdm(self.time_lut.vocab.word2idx.keys()):
                    if key == '-1':
                        time_list.append(-1)
                    else:
                        time_list.append(key[:7])
                month_time_vocab.buildVocab(time_list)
                self.time_vocab = month_time_vocab
                self.time_lut.vocab = month_time_vocab
            if time_unit == "year":
                time_list = []
                day_time_vocab = Vocabulary()
                for key in tqdm(self.time_lut.vocab.word2idx.keys()):
                    if key == -1:
                        time_list.append(-1)
                    else:
                        time_list.append(key[:4])
                day_time_vocab.buildVocab(time_list)
                self.time_vocab = day_time_vocab
                self.time_lut.vocab = day_time_vocab

        if self.graph:
            pass

    def _datable2numpy(self, data):
        """
        convert a datable to numpy array form according to the previously constructed Vocab
        :param data: datable (dataset_len,5)
        :return: numpy array
        """
        data.str2idx("head", self.node_vocab)
        data.str2idx("tail", self.node_vocab)
        data.str2idx("relation", self.relation_vocab)
        if self.time:
            if self.time_unit == "month":
                data.data["start"] = data.data.apply(lambda x: x["start"][:7], axis=1)
                data.data["end"] = data.data.apply(lambda x: x["end"][:7], axis=1)
            if self.time_unit == "year":
                data.data["start"] = data.data.apply(lambda x: x["start"][:4], axis=1)
                data.data["end"] = data.data.apply(lambda x: x["end"][:4], axis=1)
                # for index,row in tqdm(data.data.iterrows(),total=data.data.shape[0]):
                #     data.data.loc[index,"start"]=row["start"][:4]
                #     data.data.loc[index,"end"]=row["end"][:4]

        data.str2idx("start", self.time_vocab)
        data.str2idx("end", self.time_vocab)
        return data.to_numpy()

    def process(self, data):
        path = os.path.join(self.processed_path, "{}_dataset.pkl".format(data.data_type))
        if os.path.exists(path) and not self.reprocess:
            print("load {} dataset".format(data.data_type))
            with open(path, "rb") as new_file:
                new_data = pickle.loads(new_file.read())
            return new_data
        else:
            data = self._datable2numpy(data)
            if not self.time:
                data = data[:, :3]
            dataset = Cog_Dataset(data, task='kr',
                                  lookuptable_E=self.node_lut,
                                  lookuptable_R=self.relation_lut,
                                  node_type=self.nodetype,
                                  descriptions=self.description,
                                  time=self.time,
                                  relation_type=self.relationtype,
                                  )
            dataset.data_name = self.data_name
            # file = open(path, "wb")
            # file.write(pickle.dumps(dataset))
            # file.close()
            return dataset

    def process_lut(self):
        return self.node_lut, self.relation_lut, self.time_lut

