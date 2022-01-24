import torch
def graph(func):
    def inner(*args,**kwargs):
        head_embedding,relation_embedding,tail_embedding=func(*args,**kwargs)
        return head_embedding,relation_embedding,tail_embedding
    return inner