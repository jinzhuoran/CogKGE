import os


def check_make_file(file):
    if not os.path.exists(file):
        os.makedirs(file)
        print("New raw_data path has been created!The path is:", file)
    pass


def download_unzip_data(data_path, zip_name, data_url):
    if not os.path.exists(os.path.join(data_path, zip_name)):
        os.system('wget -P ' + data_path + ' ' + data_url)
        os.system("cd %s && unzip %s" % (data_path, zip_name))
    else:
        print("%s file exists in %s" % (zip_name, os.path.join(data_path, zip_name)))
    pass


def download_data(data_name, zip_name, dataset_path, url_path):
    raw_data_path = os.path.join(dataset_path,  data_name, "raw_data")
    data_url = os.path.join(url_path, "data", zip_name)
    check_make_file(raw_data_path)
    download_unzip_data(data_path=raw_data_path, zip_name=zip_name, data_url=data_url)


class Download_Data:
    def __init__(self, dataset_path):
        self.url = "http://49.232.8.218"
        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            response = input("Do you want to creat a new file?y/n\n")
            if response == "y":
                os.makedirs(self.dataset_path)
            elif response == "n":
                raise FileExistsError(self.dataset_path, "Dataset path does not exist!")
            else:
                raise ValueError("Please input y or n.")

    def FB15K(self):
        download_data(data_name="FB15K",
                      zip_name="FB15K.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)

    def FB15K237(self):
        download_data(data_name="FB15K237",
                      zip_name="FB15K237.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)

    def WN18(self):
        download_data(data_name="WN18",
                      zip_name="WN18.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)

    def WN18RR(self):
        download_data(data_name="WN18RR",
                      zip_name="WN18RR.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)

    def WIKIDATA5M(self):
        download_data(data_name="WIKIDATA5M",
                      zip_name="WIKIDATA5M.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)

    def MOBILEWIKIDATA5M(self):
        download_data(data_name="MOBILEWIKIDATA5M",
                      zip_name="MOBILEWIKIDATA5M.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)

    def EVENTKG240K(self):
        download_data(data_name="EVENTKG240K",
                      zip_name="EVENTKG240K.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)

    def CSKG(self):
        download_data(data_name="CSKG",
                      zip_name="CSKG.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)

    def FRAMENET(self):
        download_data(data_name="FRAMENET",
                      zip_name="FRAMENET.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)

    def COGNET360K(self):
        download_data(data_name="COGNET360K",
                      zip_name="COGNET360K.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)
