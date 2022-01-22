from .baseprocessor import BaseProcessor


class FB15KProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut, reprocess):
        super().__init__("FB15K", node_lut, relation_lut, reprocess)
