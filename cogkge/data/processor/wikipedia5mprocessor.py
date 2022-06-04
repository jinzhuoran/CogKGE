from .baseprocessor import BaseProcessor


class WIKIPEDIA5MProcessor(BaseProcessor):
    def __init__(self, node_lut, relation_lut):
        super().__init__(data_name="WIKIPEDIA5M",node_lut=node_lut, relation_lut=relation_lut)
