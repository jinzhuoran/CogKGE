# Preprocessing the data
# Generated files: train,valid,test triples in id form 
#                 entitiey set and relation set
# Usage:  python preprocess.py --data_path balabala  --output_path balabala
import numpy as np
import os
import pickle
import argparse


def data_generation(data_path,output_path,part):
    if part not in ['train','valid','test']:
        raise ValueError("part argument must be train,valid or test but got {}!".format(part))
    # train_file = os.path.join(data_path, 'train.txt')
    train_file = os.path.join(data_path,part+'.txt')
    id2entity_file = os.path.join(data_path, 'entities.dict')
    id2relation_file = os.path.join(data_path, 'relations.dict')

    entity2id = {}
    relation2id = {}
    entity_set = set()
    relation_set = set()
    triple_list = []

    print("Reading file {}...".format(id2entity_file))
    with open(id2entity_file, 'r') as f:
        for line in f:
            if line:
                id, entity = line.strip().split('\t')
                id = int(id)
                if entity:
                    entity2id[entity] = id
                    entity_set.add(id)

    print("Reading file {}...".format(id2relation_file))
    with open(id2relation_file, 'r') as f:
        for line in f:
            if line:
                id, relation = line.strip().split('\t')
                id = int(id)
                if relation:
                    relation2id[relation] = id
                    relation_set.add(id)

    print("Reading file {}...".format(train_file))
    with open(train_file, 'r') as f:
        for line in f:
            if line:
                h, r, t = line.strip().split('\t')
                triple_list.append([entity2id[h], relation2id[r], entity2id[t]])

    print("Saving files...")

    triples = np.array(triple_list)
    np.save(os.path.join(output_path, '{}_triples.npy'.format(part)), triples)

    with open(os.path.join(output_path, 'entity_set.pkl'.format(part)), 'wb') as f:
        pickle.dump(entity_set, f)
    with open(os.path.join(output_path, 'relation_set.pkl'.format(part)), 'wb') as f:
        pickle.dump(relation_set, f)

    print("Finished constructing {} dataset!".format(part))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FB15K37 Data Converter.')
    parser.add_argument('--data_path', default='../FB15k-237',help='choose data path to the raw dataset')
    parser.add_argument('--output_path', default='../FB15k-237',help='choose where to save the processing results')
    args = parser.parse_args()

    print("Data preprocessing in dir {}...".format(os.getcwd()))
    for p in ['train', 'test', 'valid']:
        data_generation(args.data_path,args.output_path,p)