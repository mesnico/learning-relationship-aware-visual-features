import numpy as np
from abc import ABC, abstractmethod

class OrderBase(ABC):
    def __init__(self):
        self.old_query_idx = None

    @abstractmethod
    def compute_distances(self, query_idx):
        return NotImplemented

    @abstractmethod
    def get_name():
        return NotImplemented

    def get(self, query_idx):
        #simple caching mechanism
        if query_idx != self.old_query_idx:
            self.__distances = self.compute_distances(query_idx)
            self.__ordered_distances = np.sort(self.__distances)
            self.__permuts = np.argsort(self.__distances, kind='mergesort')
            self.old_query_idx = query_idx
        return (self.__distances, self.__ordered_distances, self.__permuts)

    @abstractmethod
    def length(self):
        return NotImplemented