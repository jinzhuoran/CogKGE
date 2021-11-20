import os


class WIKIDATA5M_Filter:
    def __init__(self,
                 path_wikidata5m,
                 path_mobilewikidata5m,
                 threshold=50000):
        self.path_wikidata5m = path_wikidata5m
        self.path_mobilewikidata5m = path_mobilewikidata5m
        self.threshold = threshold
        self.entity_set=set()

    def filter_triplet(self, input_data_name, output_data_name, data_type):
        path_input = os.path.join(self.path_wikidata5m, input_data_name)
        path_output = os.path.join(self.path_mobilewikidata5m, output_data_name)
        file_output = open(path_output, "w")
        with open(path_input) as file_input:
            for line in file_input:
                if data_type == "train" or data_type == "valid":
                    h, r, t = line.strip().split("\t")
                    if int(h[1:]) < self.threshold and int(t[1:]) < self.threshold:
                        self.entity_set.add(h)
                        self.entity_set.add(t)
                        file_output.write(line)
                if data_type == "test":
                    h, r, t = line.strip().split("\t")
                    self.entity_set.add(h)
                    self.entity_set.add(t)
                    file_output.write(line)
        file_output.close()

    def filter_text(self, input_text_name, output_text_name):
        path_input = os.path.join(self.path_wikidata5m, input_text_name)
        path_output = os.path.join(self.path_mobilewikidata5m, output_text_name)
        file_output = open(path_output, "w")
        with open(path_input) as file_input:
            for line in file_input:
                text_list = line.strip().split("\t")
                if text_list[0] in self.entity_set:
                # if int(text_list[0][1:]) < self.threshold and text_list[0] in self.entity_set:

                    file_output.write(line)
        file_output.close()

    def create_MOBILEWIKIDATA5M(self):
        if not os.path.exists(self.path_wikidata5m):
            raise ValueError("Path_wikidata5m is incorrect!Please download WIKIDATA5M!")
        if not os.path.exists(self.path_mobilewikidata5m):
            os.makedirs(self.path_mobilewikidata5m)
            print(self.path_mobilewikidata5m +"\t"+'Create successfully!')
        else:
            print(self.path_mobilewikidata5m + 'The path has been existed!')
        self.filter_triplet(input_data_name="wikidata5m_transductive_train.txt",
                            output_data_name="mobilewikidata5m%s_transductive_train.txt" % (self.threshold),
                            data_type="train")
        self.filter_triplet(input_data_name="wikidata5m_transductive_valid.txt",
                            output_data_name="mobilewikidata5m%s_transductive_valid.txt" % (self.threshold),
                            data_type="valid")
        self.filter_triplet(input_data_name="wikidata5m_transductive_test.txt",
                            output_data_name="mobilewikidata5m%s_transductive_test.txt" % (self.threshold),
                            data_type="test")
        self.filter_text(input_text_name="wikidata5m_text.txt",
                         output_text_name="mobilewikidata5m%s_text" % (self.threshold))


if __name__ == '__main__':
    filter = WIKIDATA5M_Filter(path_wikidata5m="../../dataset/kr/WIKIDATA5M/raw_data",
                               path_mobilewikidata5m="../../dataset/kr/MOBILEWIKIDATA5M/raw_data",
                               threshold=50000)
    filter.create_MOBILEWIKIDATA5M()
