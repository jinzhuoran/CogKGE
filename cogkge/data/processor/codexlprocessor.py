from .baseprocessor import BaseProcessor


class CODEXLProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut, reprocess):
        super().__init__("CODEXL", node_lut, relation_lut, reprocess)
