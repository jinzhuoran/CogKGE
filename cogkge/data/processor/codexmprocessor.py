from .baseprocessor import BaseProcessor


class CODEXMProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut, reprocess):
        super().__init__("CODEXM", node_lut, relation_lut, reprocess)
