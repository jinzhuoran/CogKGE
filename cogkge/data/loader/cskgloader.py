from .baseloader import BaseLoader


class CSKGLoader(BaseLoader):
    def __init__(self, dataset_path, download=False):
        super().__init__(dataset_path, download,
                         raw_data_path="CSKG/raw_data",
                         processed_data_path="CSKG/processed_data",
                         train_name="CSKG_train.txt",
                         valid_name="CSKG_valid.txt",
                         test_name="CSKG_test.txt",
                         data_name="CSKG")

    def download_action(self):
        self.downloader.CSKG()