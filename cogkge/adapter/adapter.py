import torch
def nodetype(func):
    def inner(*args,**kwargs):
        current_device="cuda:%s"%(torch.cuda.current_device())
        head_embedding,relation_embedding,tail_embedding=func(*args,**kwargs)
        triplet_idx=args[1]
        node_lut=args[2]
        nodetype_transfer_matrix=args[3]
        embedding_dim=args[4]
        head_type=node_lut.type[triplet_idx[:, 0]].to(current_device)
        tail_type=node_lut.type[triplet_idx[:, 2]].to(current_device)
        head_embedding=torch.bmm(torch.unsqueeze(head_embedding,dim=1),nodetype_transfer_matrix(head_type).view(-1,embedding_dim,embedding_dim)).squeeze()
        tail_embedding=torch.bmm(torch.unsqueeze(tail_embedding,dim=1),nodetype_transfer_matrix(tail_type).view(-1,embedding_dim,embedding_dim)).squeeze()
        return head_embedding,relation_embedding,tail_embedding
    return inner

def description(func):
    def inner(*args,**kwargs):
        current_device="cuda:%s"%(torch.cuda.current_device())
        head_embedding,relation_embedding,tail_embedding=func(*args,**kwargs)
        triplet_idx=args[1]
        node_lut=args[2]
        pre_training_model=args[5]
        head_token=node_lut.token[triplet_idx[:,0]].to(current_device)
        tail_token=node_lut.token[triplet_idx[:,2]].to(current_device)
        head_mask=node_lut.mask[triplet_idx[:,0]].to(current_device)
        tail_mask=node_lut.mask[triplet_idx[:,2]].to(current_device)
        head_embedding=pre_training_model(head_token,attention_mask=head_mask).pooler_output
        tail_embedding=pre_training_model(tail_token,attention_mask=tail_mask).pooler_output
        return head_embedding,relation_embedding,tail_embedding
    return inner

def time(func):
    def inner(*args,**kwargs):
        current_device="cuda:%s"%(torch.cuda.current_device())
        head_embedding,relation_embedding,tail_embedding=func(*args,**kwargs)
        triplet_idx=args[1]
        time_transfer_matrix=args[6]
        start_time=time_transfer_matrix(triplet_idx[:, 3]).to(current_device)
        end_time=time_transfer_matrix(triplet_idx[:, 4]).to(current_device)
        head_embedding=torch.cat((head_embedding,start_time,end_time),dim=1)
        relation_embedding=torch.cat((relation_embedding,start_time,end_time),dim=1)
        tail_embedding=torch.cat((tail_embedding,start_time,end_time),dim=1)
        return head_embedding,relation_embedding,tail_embedding
    return inner

def graph(func):
    def inner(*args,**kwargs):
        head_embedding,relation_embedding,tail_embedding=func(*args,**kwargs)
        return head_embedding,relation_embedding,tail_embedding
    return inner