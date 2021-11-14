import os
import datetime
 

# import the specified class
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0]) 
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

# compute the output path from the data path
def cal_output_path(data_path):
    output_path = os.path.join(*data_path.split("/")[:-1], "experimental_output/" + str(datetime.datetime.now())).replace(
    ':', '-').replace(' ', '--')
    return output_path

    