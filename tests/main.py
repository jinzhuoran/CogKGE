print("Hello World")
a = 1
class Foo:
    def __init__(self,):
        pass
    def get_global(self,):
        return globals()
foo = Foo()
print(foo.get_global())