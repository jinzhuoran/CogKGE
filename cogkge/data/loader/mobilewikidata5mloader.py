import os

from .baseloader import BaseLoader
from ..lut import LookUpTable


class MOBILEWIKIDATA5MLoader(BaseLoader):
    def __init__(self, path, download=False, download_path=None):
        super().__init__(path, download, download_path,
                         train_name="mobilewikidata5m5000_transductive_train.txt",
                         valid_name="mobilewikidata5m5000_transductive_valid.txt",
                         test_name="mobilewikidata5m5000_transductive_test.txt")
        self.text_name = "mobilewikidata5m5000_text"

    def download_action(self):
        self.downloader.MOBILEWIKIDATA5M()

    def _load_lut(self, path):
        total_path = os.path.join(self.path, path).replace('\\', '/')
        lut = LookUpTable()
        lut.read_csv(total_path, sep='\t', names=["descriptions"], index_col=0)
        return lut

    def load_node_lut(self):
        pre_node_lut = os.path.join(self.path, "node_lut.pkl")
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
        return node_lut, relation_lut
