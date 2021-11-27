import os

def check_make_file(file):
    if not os.path.exists(file):
        os.makedirs(file)
        print("New dataset_path has been created!The dataset path is:", file)
    pass

def download_unzip_data(data_path,zip_name,data_url):
    if not os.path.exists(os.path.join(data_path, zip_name)):
        os.system('wget -P ' + data_path + ' ' + data_url)
        os.system("cd %s && unzip %s" % (data_path, zip_name))
    else:
        print("%s file already exists in %s" % (zip_name, os.path.join(data_path, "FB15K.zip")))
    pass

def download_data(task,data_name,zip_name,dataset_path,url_path):
    data_path = os.path.join(dataset_path, task, data_name, "raw_data")
    data_url=os.path.join(url_path,"data",zip_name)
    check_make_file(data_path)
    download_unzip_data(data_path=data_path,zip_name=zip_name,data_url=data_url)


class Download_Data:
    def __init__(self,dataset_path):
        self.url="http://49.232.8.218"
        root_path = os.getenv('HOME')
        self.dataset_path = os.path.join(root_path, dataset_path)
        if not os.path.exists(self.dataset_path):
            raise ValueError(self.dataset_path,"Dataset path is incorrect!Please enter absolute path of dataset!")
    def FB15K(self):
        download_data(task="kr",
                      data_name="FB15K",
                      zip_name="FB15K.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)
    def FB15K237(self):
        download_data(task="kr",
                      data_name="FB15K237",
                      zip_name="FB15K237.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)
    def WN18(self):
        download_data(task="kr",
                      data_name="WN18",
                      zip_name="WN18.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)
    def WN18RR(self):
        download_data(task="kr",
                      data_name="WN18RR",
                      zip_name="WN18RR.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)
    def WIKIDATA5M(self):
        download_data(task="kr",
                      data_name="WIKIDATA5M",
                      zip_name="WIKIDATA5M.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)
    def MOBILEWIKIDATA5M(self):
        download_data(task="kr",
                      data_name="MOBILEWIKIDATA5M",
                      zip_name="MOBILEWIKIDATA5M.zip",
                      dataset_path=self.dataset_path,
                      url_path=self.url)

# class Download_Visualization:
#     pass
#
# class Download_Checkpoint:
#     pass



if __name__=="__main__":
    #比如我服务器的绝对路径是/home/mentianyi（用户名）/Research_code/CogKTR/dataset
    #那么dataset_path就按如下填写
    downloader = Download_Data(dataset_path="Research_code/CogKTR/dataset")
    # downloader.FB15K()
    # downloader.FB15K237()
    # downloader.WN18()
    # downloader.WN18RR()
    downloader.WIKIDATA5M()
