import torch
class Evaluator:
    def __init__(self,
                 test_dataset,
                 model_path,
                 metric,
                 ):
        self.test_dataset=test_dataset
        self.model_path=model_path
        self.metric=metric
    def evaluate(self):
        print("The evaluating process is beginning!")
        model=torch.load(self.model_path)
        print("The model named \"%s\" has been loaded!"%(self.model_path))
        print("Model structure:\n",model)
        self.metric(self.test_dataset,model)
        print("result_rank_numpy:\n",self.metric.result_rank_numpy)
        print("mean_rank(total_sample_num_is_%d):\n"%(self.metric.sample_num),self.metric.mean_rank)
        print("hit_at_ten(total_epoch_num_is_%d):\n"%(self.metric.test_epoch),self.metric.hit_at_ten/self.metric.test_epoch*100,"%")
        print("The evaluating process is finished!")
        return 0