
def nodetype(f,type):
    def inner():
        f()
        # head_type=self.type[triplet_idx[:, 0]].to(current_device)
        # tail_type=self.type[triplet_idx[:, 2]].to(current_device)
        print("函数的执行时间为%f")
    return inner