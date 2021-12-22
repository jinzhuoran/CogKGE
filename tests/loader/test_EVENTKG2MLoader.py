from cogktr import *
import time

loader = EVENTKG2MLoader(path='/home/hongbang/CogKTR/dataset/kr/EVENTKG2M/raw_data',
                         download=True,
                         download_path="CogKTR/dataset/")

print("Without Preprocessing:")
start_time = time.time()
train_data,valid_data,test_data = loader.load_all_data()
node_lut,relation_lut,time_lut = loader.load_all_lut()
processor = EVENTKG2MProcessor(node_lut,relation_lut,time_lut)

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

