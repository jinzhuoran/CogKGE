from .baseloader import BaseLoader


class WN18Loader(BaseLoader):
    def __init__(self, path, download=False, download_path=None):
        super().__init__(path, download, download_path,
                         train_name="wordnet-mlj12-train.txt",
                         valid_name="wordnet-mlj12-valid.txt",
                         test_name="wordnet-mlj12-test.txt")

    def download_action(self):
        self.downloader.WN18()
