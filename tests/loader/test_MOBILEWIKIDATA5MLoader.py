from cogktr import *
import time

start_time = time.time()
loader = MOBILEWIKIDATA5MLoader(path='/home/hongbang/CogKTR/dataset/kr/MOBILEWIKIDATA5M/raw_data',
                                download=True,
                                download_path="CogKTR/dataset/")
train_data, valid_data, test_data = loader.load_all_data()
node_vocab, relation_vocab = loader.load_all_vocabs()
node_lut = loader.load_all_lut()

processor = MOBILEWIKIDATA5MProcessor(node_vocab,relation_vocab,node_lut)
train_dataset = processor.process(train_data)
valid_dataset = processor.process(valid_data)
test_dataset = processor.process(test_data)



for i in range(2):
    print(train_dataset[i])
    print(valid_dataset[i])
    print(test_dataset[i])
print("Train:{}  Valid:{} Test:{}".format(len(train_dataset),
                                          len(valid_dataset),
                                          len(test_dataset)))

print("--- %s seconds ---" % (time.time() - start_time))

