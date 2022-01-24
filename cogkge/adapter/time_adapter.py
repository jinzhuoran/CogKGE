import torch
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