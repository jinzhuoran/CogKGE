from cogktr import *
import time

# print("Testing FB15K dataloader...")
# loader = FB15KLoader(path='/home/hongbang/CogKTR/dataset/kr/FB15K/raw_data',
#                      download=True,
#                      download_path="CogKTR/dataset/")
# processor = FB15KProcessor()

print("Testing FB15K237 dataloader...")
loader = FB15K237Loader(path='/home/hongbang/CogKTR/dataset/kr/FB15K/raw_data',
                     download=True,
                     download_path="CogKTR/dataset/")
processor = FB15K237Processor()

# print("Testing WN18 dataloader...")
# loader = WN18Loader(path='/home/hongbang/CogKTR/dataset/kr/FB15K/raw_data',
#                     download=True,
#                     download_path="CogKTR/dataset/")
# processor = WN18Processor()
#
# print("Testing WN18RR dataloader...")
# loader = WN18RRLoader(path='/home/hongbang/CogKTR/dataset/kr/FB15K/raw_data',
#                       download=True,
#                       download_path="CogKTR/dataset/")
# processor = WN18RRProcessor()

start_time = time.time()
train_data, valid_data, test_data = loader.load_all_data()
node_vocab, relation_vocab = loader.load_all_vocabs()
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
