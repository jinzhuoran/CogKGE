from .baseloader import BaseLoader


class WN18Loader(BaseLoader):
    def __init__(self, dataset_path, download=False):
        super().__init__(dataset_path, download,
                         raw_data_path="WN18/raw_data",
                         processed_data_path="WN18/processed_data",
                         train_name="wordnet-mlj12-train.txt",
                         valid_name="wordnet-mlj12-valid.txt",
                         test_name="wordnet-mlj12-test.txt",
                         data_name="WN18")

    def download_action(self):
        self.downloader.WN18()

