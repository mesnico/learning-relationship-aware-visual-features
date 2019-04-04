import numpy as np
import os
from abc import ABC, abstractmethod
import pickle

class OrderBase(ABC):

    @abstractmethod
    def compute_distances(self, query_idx):
        return NotImplemented

    # Returns an identifier for this ordering.
    @abstractmethod
    def get_identifier():
        return NotImplemented

    # Returns name as viewed for by the user for example in graphs. Allowed '\n'.
    @abstractmethod
    def get_name():
        return NotImplemented

    def get(self, query_idx, include_query=False, min_length=0, keep_orig_consistency=False, cache_fld='DOP_cache'):
        #permutation caching mechanism
        fld_name = os.path.join(cache_fld, '{}'.format(self.get_identifier()))
        if not os.path.exists(fld_name):
            os.makedirs(fld_name)
        curr_cache_filename = os.path.join(fld_name, 'dop-{}.pkl'.format(query_idx))
        if cache_fld == None or not os.path.isfile(curr_cache_filename):
            self.__distances = self.compute_distances(query_idx)
            if min_length > 0:
                self.__distances = self.__distances[:min_length]
            self.__ordered_distances = np.sort(self.__distances)
            self.__permuts = np.argsort(self.__distances, kind='mergesort')
            with open(curr_cache_filename,'wb') as f:
                pickle.dump((self.__distances, self.__ordered_distances, self.__permuts), f)
        else:
            #load cache file
            with open(curr_cache_filename,'rb') as f:
                print('Loading cached DOPs for {}'.format(curr_cache_filename))
                self.__distances, self.__ordered_distances, self.__permuts = pickle.load(f)

        if not include_query:
            #delete first item in permuts and ordered_distances
            self.__ordered_distances = self.__ordered_distances[1:]
            self.__permuts = self.__permuts[1:]
            #delete also the query from original distance vector
            self.__distances = np.delete(self.__distances, query_idx)

            if not keep_orig_consistency:
                #adjust permutation indexes to match distance vector (from wich query has been removed)
                self.__permuts = np.asarray([n-1 if n>=query_idx else n for n in self.__permuts])
                
        return (self.__distances, self.__ordered_distances, self.__permuts)

    @abstractmethod
    def length(self):
        return NotImplemented
