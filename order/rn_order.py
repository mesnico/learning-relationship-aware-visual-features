import pickle
from . import utils
import numpy as np
from .order_base import OrderBase
from .lsh_index import build_lsh
import faiss

import pdb

'''Ordering for features extracted from the RN network'''
class RNOrder(OrderBase):
    def __init__(self, filename, name='RN', normalize=False, how_many=15000, st='test', preproc=None, **kwargs):
        super().__init__()      

        self.rn_feats = self.load_features(filename, how_many)
        self.normalize = normalize
        self.st = st
        if normalize:
            self.rn_feats = utils.normalized(self.rn_feats, 1)
        self.name = name
        self.preproc = preproc
        if preproc=='lsh':
            lsh_nbits = orig_dims // 2 if 'lsh_nbits' not in kwargs else kwargs['lsh_nbits']
            self.preproc_arg = lsh_nbits
            self.index = build_lsh(self.rn_feats, self.get_identifier(), lsh_nbits)
        if preproc=='pca':
            orig_dims = self.rn_feats.shape[1]
            pca_dims = orig_dims // 2 if 'pca_dims' not in kwargs else kwargs['pca_dims']
            self.preproc_arg = pca_dims
            mat = faiss.PCAMatrix (self.rn_feats.shape[1], pca_dims)
            mat.train(self.rn_feats)
            assert mat.is_trained
            self.rn_feats = mat.apply_py(self.rn_feats)
            assert self.rn_feats.shape[1]==pca_dims
            print('PCA from {} to {}'.format(orig_dims, pca_dims))

    def load_features(self,  filename, how_many):
        f = open(filename, 'rb')
        features = pickle.load(f)
        features = [f[1] for f in features]
        features = np.vstack(features)
        features = features[:how_many]
        print('processed #{} features each of size {}'.format(features.shape[0], features.shape[1]))
        return features
    
    def compute_distances(self, query_img_index):
        query_feat = self.rn_feats[query_img_index]
        distances = [utils.l2_dist(query_feat, f) for f in self.rn_feats]
        if self.preproc=='lsh':
            #k = entire validation set
            q = np.expand_dims(query_feat, axis=0)
            reord_distances,perm = self.index.search(q, len(self.rn_feats))

            #inverse permutation in order to reconstruct the originally ordered distances
            distances = [0] * len(perm[0])
            for d, p in zip(reord_distances[0], perm[0]):
                distances[p] = d
        return distances

    def get_name(self):
        if self.preproc != None:
            return '{}\n{}\n{}'.format(self.name, self.preproc, self.preproc_arg)
        else:
            return self.name

    def get_identifier(self):
        return '{}-norm{}-set{}'.format(self.get_name().replace('\n','_').replace(' ','-'), self.normalize, self.st)

    def length(self):
        return len(self.rn_feats)

#simple test
import os
if __name__ == "__main__":
    clevr_dir = '../features'
    idx = 6
    
    s = RNOrder(os.path.join(clevr_dir,'avg_features.pickle'))
    print(s.get(idx))
