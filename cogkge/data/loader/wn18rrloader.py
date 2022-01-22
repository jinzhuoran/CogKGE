from .baseloader import BaseLoader


class WN18RRLoader(BaseLoader):
    def __init__(self, path, download=False, download_path=None):
        super().__init__(path, download, download_path,
                         train_name="train.txt",
                         valid_name="valid.txt",
                         test_name="test.txt")

    def download_action(self):
        self.downloader.WN18RR()
