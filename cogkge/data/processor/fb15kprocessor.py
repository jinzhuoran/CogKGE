from .baseprocessor import BaseProcessor


class FB15KProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut, reprocess,train_pattern="score_based"):
        super().__init__("FB15K", node_lut=node_lut, relation_lut=relation_lut, reprocess=reprocess,train_pattern=train_pattern)
