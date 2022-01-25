from .baseprocessor import BaseProcessor


class CSKGProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut, reprocess):
        super().__init__("CSKG", node_lut, relation_lut, reprocess)