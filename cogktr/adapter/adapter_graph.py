
def graph(f):
    def inner():
        f()
        print("函数的执行时间为%f")
    return inner