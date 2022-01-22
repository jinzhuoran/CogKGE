from .baseprocessor import BaseProcessor


class FB15K237Processor(BaseProcessor):
    def __init__(self, node_lut, relation_lut, reprocess):
        super().__init__("FB15K237", node_lut, relation_lut, reprocess)
