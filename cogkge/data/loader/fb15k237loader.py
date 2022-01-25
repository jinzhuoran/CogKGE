from .baseloader import BaseLoader


class FB15K237Loader(BaseLoader):
    def __init__(self,dataset_path,download=False):
        super().__init__(dataset_path, download,
                         raw_data_path="FB15K237/raw_data",
                         processed_data_path="FB15K237/processed_data",
                         train_name="train.txt",
                         valid_name="valid.txt",
                         test_name="test.txt",
                         data_name="FB15K237"
        )
    
    def download_action(self):
        self.downloader.FB15K237()