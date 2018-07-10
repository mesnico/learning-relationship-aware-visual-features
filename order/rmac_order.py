import h5py
from . import utils
import numpy as np
from .order_base import OrderBase
import faiss

class RMACOrder(OrderBase):
    def __init__(self, rmac_file, rmac_order_filename, normalize=False, how_many=15000, st='test', preproc=None, **kwargs):
        super().__init__()

        self.st = st
        print('Loading RMAC features...')
        self.rmac_feats = self.load_rmac_features(rmac_file, rmac_order_filename, how_many)
        print('Loaded {} RMAC features ({} dims)'.format(self.rmac_feats.shape[0], self.rmac_feats.shape[1]))
        if normalize:
            self.rmac_feats = utils.normalized(self.rmac_feats, 1)
        if preproc=='pca':
            orig_dims = self.rmac_feats.shape[1]
            pca_dims = orig_dims // 2 if 'pca_dims' not in kwargs else kwargs['pca_dims']
            mat = faiss.PCAMatrix (self.rmac_feats.shape[1], pca_dims)
            mat.train(self.rmac_feats)
            assert mat.is_trained
            self.rmac_feats = mat.apply_py(self.rmac_feats)
            assert self.rmac_feats.shape[1]==pca_dims
            print('PCA from {} to {}'.format(orig_dims, pca_dims))

    def load_rmac_features(self, feat_filename, feat_order_filename,how_many):
        features = h5py.File(feat_filename, 'r')['/rmac']
        img_names = open(feat_order_filename, 'r').readlines()
        
        assert len(features) == len(img_names)

        #takes only features from a certain set
        s = 'val' if self.st=='test' else self.st
        filtered = [feat for feat, name in zip(features, img_names) if s in name]
        filtered = np.vstack(filtered)
        filtered = filtered[:how_many]
        return filtered

    def compute_distances(self, query_img_index):
        query_feat = self.rmac_feats[query_img_index]
        distances = [utils.dot_dist(query_feat, f) for f in self.rmac_feats]

        return distances

    def get_name(self):
        return 'RMAC'

    def length(self):
        return len(self.rmac_feats)

#simple test
import os
if __name__ == "__main__":
    clevr_dir = '../features'
    idx = 6

    filename = os.path.join(clevr_dir,'clevr_rmac_features.h5')
    order_filename = os.path.join(clevr_dir,'clevr_rmac_features_order.txt')
    
    s = RMACOrder(filename, order_filename)
    print(s.get(idx))
        
