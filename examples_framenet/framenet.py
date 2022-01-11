from datasets import load_dataset
import nltk
# print("start")
# nltk.download('framenet_v17')
# print("end")
from nltk.corpus import framenet as fn
frames = fn.frames()
frame_dict = {}
matrix = {}
node = {}
for frame in frames:
    ID = frame.ID
    name = frame.name
    definition = frame.definition
    lexUnit = frame.lexUnit
    FE = frame.FE
    frameRelations = frame.frameRelations
    frame_dict[ID] = {}
    frame_dict[ID]['ID'] = ID
    frame_dict[ID]['name'] = name
    frame_dict[ID]['definition'] = definition
    frame_dict[ID]['lu'] = {}
    frame_dict[ID]['fe'] = {}
    if ID not in node:
        node[ID] = {}
        node[ID]['name'] = name
        node[ID]['definition'] = definition
    if ID not in matrix:
        matrix[ID] = {}
    for lu in lexUnit.values():
        frame_dict[ID]['lu'][lu.name] = lu.definition
        matrix[ID][lu.ID] = 'lu'
        if lu.ID not in node:
            node[lu.ID] = {}
            node[lu.ID]['name'] = lu.name
            node[lu.ID]['definition'] = lu.definition
    for fe in FE.values():
        frame_dict[ID]['fe'][fe.name] = fe.definition
        matrix[ID][fe.ID] = 'fe'
        if fe.ID not in node:
            node[fe.ID] = {}
            node[fe.ID]['name'] = fe.name
            node[fe.ID]['definition'] = fe.definition
    for fr in frameRelations:
        subID = fr.subID
        supID = fr.supID
        if subID not in matrix:
            matrix[subID] = {}
            matrix[subID][supID] = fr.type['name']
print("finish")