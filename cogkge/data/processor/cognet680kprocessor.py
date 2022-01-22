from .baseprocessor import BaseProcessor


class COGNET680KProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut):
        super().__init__("COGNET680K",node_lut, relation_lut)
