from .baseprocessor import BaseProcessor


class COGNET360KProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut,reprocess=True,train_pattern="score_based"):
        super().__init__("COGNET360K", node_lut, relation_lut,reprocess=reprocess,train_pattern=train_pattern)
