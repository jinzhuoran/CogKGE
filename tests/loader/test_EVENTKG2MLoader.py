from cogktr import *
import time

loader = EVENTKG2MLoader(path='/home/hongbang/CogKTR/dataset/kr/EVENTKG2M/raw_data',
                         download=True,
                         download_path="CogKTR/dataset/")
processor = EVENTKG2MProcessor()
print("Without Preprocessing:")
start_time = time.time()
train_data,valid_data,test_data = loader.load_all_data()
node_vocab,relation_vocab,time_vocab = loader.load_all_vocabs()
node_lut,relation_lut = loader.load_all_lut()
train_dataset = processor.process(train_data)

for i in range(10):
    print(train_dataset[i])

print("--- %s seconds ---" % (time.time() - start_time))

