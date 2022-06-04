from .baseloader import BaseLoader


class WIKIPEDIA5MLoader(BaseLoader):
    def __init__(self, dataset_path, download=False):
        super().__init__(dataset_path, download,
                         raw_data_path="WIKIPEDIA5M/raw_data",
                         processed_data_path="WIKIPEDIA5M/processed_data",
                         train_name="processed_wikidata5m_transductive_train.txt",
                         valid_name="processed_wikidata5m_transductive_valid.txt",
                         test_name="processed_wikidata5m_transductive_test.txt",
                         data_name="WIKIPEDIA5M")

    def download_action(self):
        pass

