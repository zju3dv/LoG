import numpy as np
import torch

class IterationBasedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, iterations, index=None):
        if index is None:
            self.index = np.arange(len(dataset))
        else:
            print(f'[{self.__class__.__name__}] manual set index {len(index)}')
            self.index = index
        self.iterations = iterations
    
    def __len__(self):
        return self.iterations
    
    def __iter__(self):
        for i in range(self.iterations):
            yield np.random.choice(self.index)

class IndexSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, index=None):
        if index is not None:
            self.index = index
        else:
            self.index = np.arange(len(dataset))
    
    def __len__(self):
        return len(self.index)
    
    def __iter__(self):
        return iter(self.index)