from .baseprocessor import BaseProcessor


class WN18RRProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut):
        super().__init__(node_lut, relation_lut)
