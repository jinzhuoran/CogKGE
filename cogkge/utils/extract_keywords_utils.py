import re
import os
from tqdm import tqdm

class Extract_Keywords():
    def __init__(self,raw_data_path,new_data_path,raw_data_name):
        self.raw_data_name=raw_data_name
        self.raw_data_path=os.path.join(raw_data_path,self.raw_data_name)
        self.new_data_path=os.path.join(new_data_path,self.raw_data_name)[:-3]+".txt"
        self.re_dict={
            "type_labels_dbpedia.nq":{
                "instance":r"<http://dbpedia.org/ontology/(.*?)>",
                "link":r"<http://www.w3.org/2000/01/rdf-schema#(.*?)>",
                "value":r'"(.*?)"@en '
            }
        }
        self.raw_data_length=0
        self.new_data_length=0
    def start(self):
        file_new=open(self.new_data_path,"w")
        print("Extracting...")
        with open(self.raw_data_path,"r", encoding='gb18030') as file:
            for line in tqdm(file):
                line=line.strip()
                self.raw_data_length=self.raw_data_length+1
                re_instance=self.re_dict[self.raw_data_name]["instance"]
                re_link=self.re_dict[self.raw_data_name]["link"]
                re_value=self.re_dict[self.raw_data_name]["value"]
                match_instance=re.findall(re_instance,line)
                match_link=re.findall(re_link,line)
                match_value=re.findall(re_value,line)
                if len(match_instance)!=0 and len(match_link)!=0 and len(match_value)!=0:
                    new_line=match_instance[0]+"\t"+match_link[0]+"\t"+match_value[0]+"\n"
                    file_new.write(new_line)
                    self.new_data_length=self.new_data_length+1
        file_new.close()
        print("Extract success!")
    def show_details(self,num=5):
        print("new_data_path:",self.new_data_path)
        print("raw_data_length:",self.raw_data_length)
        print("new_data_length:",self.new_data_length)
        print("Show the first %d lines:"%(num))
        with open(self.new_data_path,"r") as file:
            for count,line in enumerate(file):
                if count>num:
                    break
                else:
                    line=line.strip()
                    print(line)



if __name__=="__main__":
    #raw_data_name=["type_labels_dbpedia.nq"]
    extract_keywords=Extract_Keywords(raw_data_path="",
                                      new_data_path="",
                                      raw_data_name="type_labels_dbpedia.nq")
    extract_keywords.start()
    extract_keywords.show_details()