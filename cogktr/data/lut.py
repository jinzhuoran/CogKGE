class LUT:
    def __init__(self,entity2id,relation2id):
        """
        entity2id:entity names -> entity id  dictionary
        relation2id:relation names -> relation id   dictionary
        """
        self.entity2id_ = entity2id
        self.relation2id_= relation2id
        self.id2entity_ = {value:key for key,value in entity2id.items()}
        self.id2relation_ = {value:key for key,value in relation2id.items()}

    def entity2id(self,entity):
        return self.entity2id_[entity]

    def id2entity(self,id):
        return self.id2entity_[id]
    
    def relation2id(self,relation):
        return self.relation2id_[relation]

    def id2relation(self,id):
        return self.id2relation_[id]
    
    def num_entity(self):
        return len(self.entity2id_)
    
    def num_relation(self):
        return len(self.relation2id_)
    

