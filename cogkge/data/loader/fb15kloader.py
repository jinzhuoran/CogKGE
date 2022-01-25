from .baseloader import BaseLoader


class FB15KLoader(BaseLoader):
    def __init__(self, dataset_path, download=False):
        super().__init__(dataset_path, download,
                         raw_data_path="FB15K/raw_data",
                         processed_data_path="FB15K/processed_data",
                         train_name="freebase_mtr100_mte100-train.txt",
                         valid_name="freebase_mtr100_mte100-valid.txt",
                         test_name="freebase_mtr100_mte100-test.txt",
                         data_name="FB15K")

    def download_action(self):
        self.downloader.FB15K()