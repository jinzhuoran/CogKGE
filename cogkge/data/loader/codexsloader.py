from .baseloader import BaseLoader


class CODEXSLoader(BaseLoader):
    def __init__(self, dataset_path, download=False):
        super().__init__(dataset_path, download,
                         raw_data_path="CODEXS/raw_data",
                         processed_data_path="CODEXS/processed_data",
                         train_name="train.txt",
                         valid_name="valid.txt",
                         test_name="test.txt",
                         data_name="CODEXS")

    def download_action(self):
        self.downloader.CODEXS()