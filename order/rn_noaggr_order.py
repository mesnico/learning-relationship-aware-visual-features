import pickle
import os
import re
from scipi import spatial.distance
from . import utils
import numpy as np
from .order_base import OrderBase
import pdb

class RNFuzzyLoader():
    def __init__(self, folder):
        self.files = ordered_file_list(folder)
        #simple pool where swap is performed by last accessed policy
        self.loaded_items = []
        self.max_loaded_items = 2
        self.old_file_id = -1
        self.folder = folder

    def ordered_file_list(self, folder):
        def natural_keys(filename):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [ atoi(c) for c in re.split('(\d+)', filename) ]
        files = os.listdir(folder)
        files.sort(key=natural_keys)
        pdb.set_trace()
        return files

    def __get_item__(self, idx):    #TODO: check
        #loaded_item_id = len(self.loaded_items) if len(self.loaded_items) < self.max_loaded_items - 1 else self.max_loaded_items - 1
        #load the idx element from the file it is contained in
        file_id = idx / 300
        row = idx % 300
        filename = self.files[file_id]

        if len(self.loaded_items) > file_id and file_id == self.old_file_id:
            actually_loaded = self.loaded_items[loaded_item_id]
        else:
            with open(os.path.join(self.folder,filename), 'rb') as f:
                actually_loaded = pickle.load(f)[row]
                self.loaded_items[-1] = actually_loaded

        self.old_file_id = file_id
        assert actually_loaded[0] == idx, "idx doesn't match!"
        return actually_loaded[1]

    def __len(self):
        #TODO: not exact length
        return 300*(len(self.files) - 1)
        
'''Ordering for features extracted from the RN network'''
class RNFuzzyOrder(OrderBase):
    def __init__(self, folder, name='RN\nFuzzy', ncpu=4, normalize=False):
        super().__init__()
        self.folder = folder
        self.ncpu = 4
        #if normalize:
        #    self.rn_feats = utils.normalized(self.rn_feats, 1)
        self.name = name

    def _compute_distance(self, rels1, rels2):
        def similarity(relations_scene1, relations_scene2):
            sim = sum([ max([distance.cosine(rel1, rel2) for rel2 in relations_scene2]) for rel1 in relations_scene1])
            return sim
        
        symm_sim = min(similarity(rels1, rels2), similarity(rels2, rels1))
        dist = 1 - symm_sim
        return dist
    
    def compute_distances(self, query_img_index):
        loader = RNFuzzyLoader(self.folder)
        return parallel_distances('rn_states-{}'.format(self.mode), loader, query_img_index, self._compute_distance, ncpu=self.ncpu)

        return distances

    def get_name(self):
        return self.name

    def length(self):
        return len(self.)

#simple test
import os
if __name__ == "__main__":
    clevr_dir = '../features'
    idx = 6
    
    s = RNOrder(os.path.join(clevr_dir,'avg_features.pickle'))
    print(s.get(idx))
