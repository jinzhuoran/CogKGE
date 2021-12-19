import math
import copy
import torch
import numpy as np
from tqdm import tqdm
import prettytable as pt
import torch.utils.data as Data
from collections import defaultdict

class Link_Prediction(object):
    def __init__(self,
                 batch_size,
                 reverse,
                 link_prediction_raw=True,
                 link_prediction_filt=False):
        self.batch_size=batch_size
        self.reverse=reverse
        self.link_prediction_raw=link_prediction_raw
        self.link_prediction_filt=link_prediction_filt

        self._metric_result_list =list()
        self._nodetype_columnindex_dict={"head":0,"tail":2}
        self.create_correct_triplet_dict_flag=True

    def _init_cogkr_metric(self):
        torch.cuda.empty_cache()
        self._raw_rank_list = list()
        self._filt_rank_list = list()
        self._raw_rank=-1
        self._filt_rank=-1
        self._Raw_MR = -1
        self._Raw_MRR = -1
        self._Raw_Hits1 = -1
        self._Raw_Hits3 = -1
        self._Raw_Hits10 = -1
        self._Filt_MR =-1
        self._Filt_MRR = -1
        self._Filt_Hits1 = -1
        self._Filt_Hits3 = -1
        self._Filt_Hits10 = -1

    def _create_correct_node_dict(self,dataset):
        node_dict=defaultdict(dict)
        for index in range(len(dataset)):
            r_t=tuple(dataset.data[index][1:])
            h_r=tuple(dataset.data[index][:2])
            h=torch.tensor(dataset.data[index][0])
            t=torch.tensor(dataset.data[index][2])
            if r_t not in node_dict["head"]:
                node_dict["head"].setdefault(r_t,[])
            node_dict["head"][r_t].append(h)
            if h_r not in node_dict["tail"]:
                node_dict["tail"].setdefault(h_r,[])
            node_dict["tail"][h_r].append(t)
        return node_dict
    def _init_correct_node_dict(self):
        if self._train_dataset!=None:
            self._correct_train_node_dict=self._create_correct_node_dict(self._train_dataset)
        if self._valid_dataset!=None:
            self._correct_valid_node_dict=self._create_correct_node_dict(self._valid_dataset)
        if self._test_dataset!=None:
            self._correct_test_node_dict=self._create_correct_node_dict(self._test_dataset)
        self.create_correct_triplet_dict_flag=False



    def _generate_expanded_triplet(self,single_triplet,entity_type):
        # (1,3,self._outer_batch_size)
        expanded_triplet=single_triplet.expand(self._node_dict_len, 3,self._single_batch_len)
        # (self._node_dict_len,3,self._single_batch_len)
        if entity_type=="head":
            original_r_t = expanded_triplet[:, 1:,:]
            expanded_node =  torch.unsqueeze(torch.unsqueeze(torch.arange(0,self._node_dict_len), dim=1), dim=1).expand(self._node_dict_len, 1,self._single_batch_len).to(self._device)
            expanded_triplet = torch.cat([expanded_node, original_r_t], dim=1)
        if entity_type=="tail":
            original_h_r=expanded_triplet[:, :2,:]
            expanded_node= torch.unsqueeze(torch.unsqueeze(torch.arange(0, self._node_dict_len), dim=1), dim=1).expand(self._node_dict_len, 1,self._single_batch_len).to(self._device)
            expanded_triplet= torch.cat([original_h_r,expanded_node], dim=1)
        return expanded_triplet

    def _caculate_rank(self,single_sample,entity_type):
        single_triplet=copy.deepcopy(single_sample)   #(1,3,self._outer_batch_size)
        column_index=self._nodetype_columnindex_dict[entity_type]
        correct_id=single_triplet[:,column_index]
        expanded_triplet=self._generate_expanded_triplet(single_triplet,entity_type)

        expanded_triplet=expanded_triplet.permute(2,0,1)
        expanded_triplet= torch.reshape(expanded_triplet,(-1,3))

        score_list = list()
        for i in range(math.ceil(self._node_dict_len / self.batch_size)):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, self._node_dict_len*self._single_batch_len)
            data_input = expanded_triplet[start:end, :]
            with torch.no_grad():
                result=self._model(data_input)
                score_list.append(result)
        sorted_score_rank= torch.argsort(torch.cat(score_list,dim=0).reshape(self._single_batch_len,-1).T,dim=0)  #torch.Size([14541, batch])
        # len_b=self.batch_size if self.batch_size<len(sorted_score_rank) else len(sorted_score_rank)
        len_b=len(sorted_score_rank)
        correct_triplet_rank=torch.mm(torch.unsqueeze(torch.arange(0,len_b),dim=0),(sorted_score_rank==correct_id).type(torch.LongTensor))[0]+1

        if self.link_prediction_raw:
            raw_correct_triplet_rank=self._node_dict_len+1-correct_triplet_rank if self.reverse else correct_triplet_rank
            self._raw_rank_list.append(raw_correct_triplet_rank.tolist())


        if self.link_prediction_filt:
            arr=1 if entity_type=="head" else 0
            sorted_score_rank=sorted_score_rank.cpu()
            triplet_key_list=list(tuple(single_triplet[0][arr:arr+2,:].permute(1,0).cpu().numpy()[i]) for i in range(len(single_triplet[0][arr:arr+2,:].permute(1,0))))
            for i,triplet_key in enumerate(triplet_key_list):
                valid_correct_node_index_list=[]
                train_correct_node_index_list=[]
                test_correct_node_index_list=[]
                if triplet_key in self._correct_valid_node_dict[entity_type].keys():
                    valid_correct_node_index_list=torch.where(sorted_score_rank[:,i] ==torch.unsqueeze(torch.tensor(self._correct_valid_node_dict[entity_type][triplet_key]),dim=1))[1]+1
                if triplet_key in self._correct_train_node_dict[entity_type].keys():
                    train_correct_node_index_list=torch.where(sorted_score_rank[:,i] == torch.unsqueeze(torch.tensor(self._correct_train_node_dict[entity_type][triplet_key]),dim=1))[1]+1
                if self._test_dataset!=None:
                    if triplet_key in self._correct_test_node_dict[entity_type].keys():
                        test_correct_node_index_list=torch.where(sorted_score_rank[:,i] == torch.unsqueeze(torch.tensor(self._correct_test_node_dict[entity_type][triplet_key]),dim=1))[1]+1
                valid_correct_before_num=torch.sum(torch.as_tensor(valid_correct_node_index_list)<correct_triplet_rank[i])
                train_correct_before_num=torch.sum(torch.as_tensor(train_correct_node_index_list)<correct_triplet_rank[i])
                if self._test_dataset!=None:
                    test_correct_before_num=torch.sum(torch.as_tensor(test_correct_node_index_list)<correct_triplet_rank[i])
                else:
                    test_correct_before_num=0
                correct_triplet_num=valid_correct_before_num+train_correct_before_num+test_correct_before_num
                filt_correct_triplet_rank=correct_triplet_rank[i]-correct_triplet_num
                self._filt_rank_list.append(filt_correct_triplet_rank)



    def caculate(self,
                 device,
                 model,
                 total_epoch,
                 current_epoch,
                 metric_type,
                 metric_dataset,
                 node_dict_len,
                 model_name,
                 logger=None,
                 writer=None,
                 train_dataset=None,
                 valid_dataset=None,
                 test_dataset=None):
        self._device=device
        self._model=model
        self._total_epoch=total_epoch
        self._current_epoch=current_epoch
        self._metric_type=metric_type
        self._metric_dataset=metric_dataset
        self._node_dict_len=node_dict_len
        self._model_name=model_name
        self._logger=logger
        self._writer=writer
        self._train_dataset=train_dataset
        self._valid_dataset=valid_dataset
        self._test_dataset=test_dataset
        Raw_Result_list=list()
        Filt_Result_list=list()

        self._init_cogkr_metric()
        if self.link_prediction_filt and self.create_correct_triplet_dict_flag:
            self._init_correct_node_dict()

        self._outer_batch_size=max(math.floor(self.batch_size/len(metric_dataset)),1)

        metric_loader = Data.DataLoader(dataset=metric_dataset, batch_size=self._outer_batch_size, shuffle=False)
        for step, single_sample in enumerate(tqdm(metric_loader)):
            self._single_batch_len=len(single_sample)     #(self._outer_batch_size,3)
            single_sample=single_sample.permute(1,0)      #(3,self._outer_batch_size)
            single_sample=torch.unsqueeze(single_sample,dim=0)  #(1,3,self._outer_batch_size)
            single_sample=single_sample.to(self._device)
            self._caculate_rank(single_sample,"tail")
            self._caculate_rank(single_sample,"head")
        if self.link_prediction_raw:
            self._raw_rank_list=sum(self._raw_rank_list, [])
            self._raw_rank=torch.tensor(self._raw_rank_list,dtype= torch.float64)
            self._Raw_Hits1 = round((torch.sum(self._raw_rank <= 1) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._Raw_Hits3  = round((torch.sum(self._raw_rank <= 3) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._Raw_Hits10 = round((torch.sum(self._raw_rank<= 10) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._Raw_MR= round(torch.mean(self._raw_rank).item(), 3)
            self._Raw_MRR  = round(torch.mean(1/self._raw_rank).item(), 3)
            Raw_Result_list=[self._Raw_Hits1,self._Raw_Hits3,self._Raw_Hits10,self._Raw_MR,self._Raw_MRR]
        if self.link_prediction_filt:
            self._filt_rank=torch.tensor(self._filt_rank_list,dtype= torch.float64)
            self._Filt_Hits1 = round((torch.sum(self._filt_rank <= 1) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._Filt_Hits3  = round((torch.sum(self._filt_rank <= 3) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._Filt_Hits10 = round((torch.sum(self._filt_rank<= 10) / (2 * len(metric_dataset)) * 100).item(), 3)
            self._Filt_MR= round(torch.mean(self._filt_rank).item(), 3)
            self._Filt_MRR  = round(torch.mean(1/self._filt_rank).item(), 3)
            Filt_Result_list=[self._Filt_Hits1,self._Filt_Hits3,self._Filt_Hits10,self._Filt_MR,self._Filt_MRR]
        if self.link_prediction_raw or self.link_prediction_filt:
            self._metric_result_list.append([self._current_epoch]+Raw_Result_list+Filt_Result_list)


    def get_Raw_Rank(self):
        return self._raw_rank
    def get_Filt_Rank(self):
        return self._filt_rank
    def get_Raw_MR(self):
        return self._Raw_MR
    def get_Raw_MRR(self):
        return self._Raw_MRR
    def get_Raw_Hits1(self):
        return self._Raw_Hits1
    def get_Raw_Hits3(self):
        return self._Raw_Hits3
    def get_Raw_Hits10(self):
        return self._Raw_Hits10
    def get_Filt_MR(self):
        return self._Filt_MR
    def get_Filt_MRR(self):
        return self._Filt_MRR
    def get_Filt_Hits1(self):
        return self._Filt_Hits1
    def get_Filt_Hits3(self):
        return self._Filt_Hits3
    def get_Filt_Hits10(self):
        return self._Filt_Hits10
    def print_current_table(self):
        tb = pt.PrettyTable()
        tb.field_names = [self._model_name,
                          "Data",
                          "Epoch/Total",
                          "Hits@1",
                          "Hits@3",
                          "Hits@10",
                          "MR",
                          "MRR"]
        if self.link_prediction_raw:
            tb.add_row(["Raw",
                        self._metric_type,
                        str(self._current_epoch)+"/"+str(self._total_epoch),
                        self._Raw_Hits1,
                        self._Raw_Hits3,
                        self._Raw_Hits10,
                        self._Raw_MR,
                        self._Raw_MRR])
        if self.link_prediction_filt:
            tb.add_row(["Filt",
                        self._metric_type,
                        str(self._current_epoch)+"/"+str(self._total_epoch),
                        self._Filt_Hits1,
                        self._Filt_Hits3,
                        self._Filt_Hits10,
                        self._Filt_MR,
                        self._Filt_MRR])
        print(tb)

    def print_best_table(self,front=3,key="Raw_MR"):
        front=len(self._metric_result_list) if len(self._metric_result_list)<front else front
        strat_index=5 if self.link_prediction_raw and self.link_prediction_filt else 0
        type_dict={"Raw_Hits@1":[1,True],
                   "Raw_Hits@3":[2,True],
                   "Raw_Hits@10":[3,True],
                   "Raw_MR":[4,False],
                   "Raw_MRR":[5,True],
                   "Filt_Hits@1":[1+strat_index,True],
                   "Filt_Hits@3":[2+strat_index,True],
                   "Filt_Hits@10":[3+strat_index,True],
                   "Filt_MR":[4+strat_index,False],
                   "Filt_MRR":[5+strat_index,True]}
        table_title=[self._model_name,
                     "Last",
                     "Best",
                     "2nd",
                     "3rd",
                     "4th",
                     "5th"]
        tb = pt.PrettyTable()
        tb.field_names = table_title[:front+2]
        last_result=self._metric_result_list[-1]
        self._metric_result_list.sort(key=lambda x: x[type_dict[key][0]], reverse=type_dict[key][1])
        self._metric_result_list=[last_result]+self._metric_result_list
        result_list_T=np.array(self._metric_result_list).T.tolist()
        table_row_title=list()
        raw_table_row_title=list()
        filt_table_row_title=list()
        if self.link_prediction_raw:
            raw_table_row_title=["Raw_Hits@1","Raw_Hits@3","Raw_Hits@10","Raw_MR","Raw_MRR"]
        if self.link_prediction_filt:
            filt_table_row_title=["Filt_Hits@1","Filt_Hits@3","Filt_Hits@10","Filt_MR","Filt_MRR"]
        if self.link_prediction_raw or self.link_prediction_filt:
            table_row_title=["Epoch"]+raw_table_row_title+filt_table_row_title
        for i in range(len(table_row_title)):
            tb.add_row([table_row_title[i]]+result_list_T[i][:front+1])
        print(tb)
        if self.link_prediction_raw:
            self._logger.info("Best: Epoch {}  Raw_Hits@1:{}   Raw_Hits@3:{}   Raw_Hits@10:{}   Raw_MR:{}   Raw_MRR:{}".format(
                self._metric_result_list[1][0],
                self._metric_result_list[1][1],
                self._metric_result_list[1][2],
                self._metric_result_list[1][3],
                self._metric_result_list[1][4],
                self._metric_result_list[1][5]))
        if self.link_prediction_filt:
            self._logger.info("Best: Epoch {}  Filt_Hits@1:{}   Filt_Hits@3:{}   Filt_Hits@10:{}   Filt_MR:{}   Filt_MRR:{}   ".format(
                self._metric_result_list[1][0],
                self._metric_result_list[1][1+strat_index],
                self._metric_result_list[1][2+strat_index],
                self._metric_result_list[1][3+strat_index],
                self._metric_result_list[1][4+strat_index],
                self._metric_result_list[1][5+strat_index]))

    def log(self):
        if self.link_prediction_raw:
            self._logger.info("{} Epoch {}/{}  Raw_Hits@1:{}   Raw_Hits@3:{}   Raw_Hits@10:{}   Raw_MR:{}   Raw_MRR:{}".format(
                self._metric_type,self._current_epoch, self._total_epoch, self._Raw_Hits1, self._Raw_Hits3, self._Raw_Hits10, self._Raw_MR, self._Raw_MRR))
        if self.link_prediction_filt:
            self._logger.info("{} Epoch {}/{}  Filt_Hits@1:{}   Filt_Hits@3:{}   Filt_Hits@10:{}   Filt_MR:{}   Filt_MRR:{}   ".format(
                self._metric_type,self._current_epoch, self._total_epoch, self._Filt_Hits1, self._Filt_Hits3, self._Filt_Hits10, self._Filt_MR, self._Filt_MRR))
    def write(self):
        if self._writer!=None:
            if self.link_prediction_raw:
                self._writer.add_scalars("Hits@1", {"{}_Raw_Hits@1".format(self._metric_type): self._Raw_Hits1}, self._current_epoch)
                self._writer.add_scalars("Hits@3", {"{}_Raw_Hits@3".format(self._metric_type): self._Raw_Hits3}, self._current_epoch)
                self._writer.add_scalars("Hits@10", {"{}_Raw_Hits@10".format(self._metric_type): self._Raw_Hits10}, self._current_epoch)
                self._writer.add_scalars("MR", {"{}_Raw_MR".format(self._metric_type): self._Raw_MR}, self._current_epoch)
                self._writer.add_scalars("MRR", {"{}_Raw_MRR".format(self._metric_type): self._Raw_MRR}, self._current_epoch)
            if self.link_prediction_filt:
                self._writer.add_scalars("Hits@1", {"{}_Filt_Hits@1".format(self._metric_type): self._Filt_Hits1}, self._current_epoch)
                self._writer.add_scalars("Hits@3", {"{}_Filt_Hits@3".format(self._metric_type): self._Filt_Hits3},self._current_epoch)
                self._writer.add_scalars("Hits@10", {"{}_Filt_Hits@10".format(self._metric_type): self._Filt_Hits10},self._current_epoch)
                self._writer.add_scalars("MR", {"{}_Filt_MR".format(self._metric_type): self._Filt_MR}, self._current_epoch)
                self._writer.add_scalars("MRR", {"{}_Filt_MRR".format(self._metric_type): self._Filt_MRR}, self._current_epoch)

if __name__ == "__main__":
    import torch.nn as nn

    #test_triples
    #node_len=4,relation_len=3,data_len=5
    #data_size=(5,3)
    test_triples=torch.tensor([[3,2,1],
                               [0,2,3],
                               [3,0,2],
                               [3,0,1],
                               [0,1,1]])

    #Test_Model
    #score=0.1*(h+r+t)
    #input_size=(data_len,3)
    #output_size=(data_len)
    class Test_Model(nn.Module):
        def __init__(self):
            super(Test_Model, self).__init__()

        def forward(self,triplet_idx):
            score=-triplet_idx[:, 2]*0.1
            return score
    test_model=Test_Model()
    score=test_model(test_triples)  #(batch,)
    print(score)

    #metric设置的，batch_size=11
    #outer_batch_size=floor(11/node_len)=2
    #fake_triples size=(data_len,3)-->metric_triples size=(outer_batch_size*node_len=8,3)
    metric = Link_Prediction(link_prediction_raw=True,
                             link_prediction_filt=True,
                             batch_size=11,
                             reverse=False)
    metric.caculate(device="cuda:0",
                    model=test_model,
                    total_epoch=10,
                    current_epoch=1,
                    metric_type="valid_dataset",
                    metric_dataset=test_triples,
                    node_dict_len=4,
                    model_name="Test_Model",
                    train_dataset=test_triples.numpy(),
                    valid_dataset=test_triples.numpy())
    metric.print_current_table()

