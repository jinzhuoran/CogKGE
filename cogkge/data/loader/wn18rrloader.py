from .baseloader import BaseLoader


class WN18RRLoader(BaseLoader):
    def __init__(self, dataset_path, download=False):
        super().__init__(dataset_path, download,
                         raw_data_path="WN18RR/raw_data",
                         processed_data_path="WN18RR/processed_data",
                         train_name="train.txt",
                         valid_name="valid.txt",
                         test_name="test.txt",
                         data_name="WN18RR")

    def download_action(self):
        self.downloader.WN18RR()