import pickle
from . import utils
import numpy as np
from .order_base import OrderBase

'''Ordering for features extracted from the RN network'''
class RNOrder(OrderBase):
    def __init__(self, filename, name='RN', normalize=False):
        super().__init__()
        self.rn_feats = self.load_features(filename)
        if normalize:
            self.rn_feats = utils.normalized(self.rn_feats, 1)
        self.name = name

    '''def load_features(self,  filename):
        f = open(filename, 'rb')
        features = pickle.load(f)
        features = [f[1] for f in features]
        features = np.vstack(features)
        print('processed #{} features each of size {}'.format(features.shape[0], features.shape[1]))
        return features'''

    def load_features(self,  filename):
        f = open(filename, 'rb')
        features = pickle.load(f)
        #features = [f[1] for f in features]
        #features = np.vstack(features)
        print('processed #{} features each of size {}'.format(features.shape[0], features.shape[1]))
        return features
    
    def compute_distances(self, query_img_index):
        query_feat = self.rn_feats[query_img_index]
        distances = [utils.l2_dist(query_feat, f) for f in self.rn_feats]

        return distances

    def get_name(self):
        return self.name

    def length(self):
        return len(self.rn_feats)

#simple test
import os
if __name__ == "__main__":
    clevr_dir = '../features'
    idx = 6
    
    s = RNOrder(os.path.join(clevr_dir,'avg_features.pickle'))
    print(s.get(idx))
