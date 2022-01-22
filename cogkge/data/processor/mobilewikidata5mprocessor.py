from .baseprocessor import BaseProcessor
from transformers import RobertaTokenizer
from transformers import RobertaModel
from tqdm import tqdm
import torch
from ..dataset import Cog_Dataset


class MOBILEWIKIDATA5MProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut):
        """
        :param luts:node_lut,relation_lut
        """
        super().__init__(node_lut, relation_lut)
        self.node_lut = node_lut
        self.pre_training_model_name = "roberta-base"
        self.token_length = 10
        self.tokenizer = RobertaTokenizer.from_pretrained(self.pre_training_model_name)
        self.pre_training_model = RobertaModel.from_pretrained(self.pre_training_model_name)

    def process(self, data):
        tokens_list = []
        masks_list = []
        for i in tqdm(range(len(self.node_lut))):
            encoded_text = self.tokenizer.encode_plus(
                self.node_lut["descriptions"][i],
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

        head_input_ids = []
        head_attention_mask = []
        tail_intput_ids = []
        tail_attention_mask = []
        for i in tqdm(range(len(data))):
            head, tail = data[i]["head"], data[i]["tail"]
            head_input_ids.append(self.node_lut.search(head, "input_ids"))
            head_attention_mask.append(self.node_lut.search(head, "attention_mask"))
            tail_intput_ids.append(self.node_lut.search(tail, "input_ids"))
            tail_attention_mask.append(self.node_lut.search(tail, "attention_mask"))
        descriptions = [torch.cat(l, dim=0) for l in
                        [head_input_ids, tail_intput_ids, head_attention_mask, tail_attention_mask]]

        data = self._datable2numpy(data)

        return Cog_Dataset(data, task='kr', descriptions=descriptions)
