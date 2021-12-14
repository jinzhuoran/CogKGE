from ...dataset import Cog_Dataset


class BaseProcessor:
    def __init__(self):
        pass

    def process(self, data):
        return Cog_Dataset(data, task='kr')
