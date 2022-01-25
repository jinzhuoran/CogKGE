from .baseprocessor import BaseProcessor


class CODEXSProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut, reprocess):
        super().__init__("CODEXS", node_lut, relation_lut, reprocess)
