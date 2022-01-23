class Vocabulary:
    def __init__(self, ):
        self.word2idx = {}
        self.idx2word = {}

    def buildVocab(self, tokens, *args):
        """
        build word2idx and idx2name based on a list of tokens
        :param tokens:[token1,token2,...]
        """
        for item in [tokens, *args]:
            for token in item:
                if token not in self.word2idx:
                    self.word2idx[token] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def getWord2idx(self):
        return self.word2idx

    def getIdx2word(self):
        return self.idx2word

    # def word2idx(self,word):
    #     return self.word2idx[word]
    #
    # def idx2word(self,idx):
    #     return self.idx2word[idx]

    def __len__(self):
        return len(self.word2idx)
