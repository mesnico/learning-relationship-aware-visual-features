import matplotlib
matplotlib.use('Agg')    
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import os
import argparse
import numpy as np
from order import rn_order, rmac_order, graphs_order, graphs_approx_order, states_order, graphs_approx_order
from tqdm import tqdm, trange
import math
import json

MAX_OBJS = 5

class ClevrImageLoader():
    def __init__(self, images_dir, st):
        self.images_dir = images_dir
        self.st = st

    def get(self,index):
        padded_index = str(index).rjust(6,'0')
        s = 'val' if self.st=='test' else self.st
        img_filename = 'CLEVR_{}_{}.png'.format(s, padded_index)
        img_fullname = os.path.join(self.images_dir, s, img_filename)
        image = Image.open(img_fullname).convert('RGB')
        return image

def build_figure(orders, image_loader, query_idx, n=10, scale=1, indexes=None):
    size = (math.ceil(4*n*scale), math.ceil(3*(len(orders)+1)*scale)+3)

    fig = plt.figure('Query idx {}'.format(query_idx), figsize=size)
    gs = gridspec.GridSpec(len(orders)+1, 1)

    #query_img
    query_axs = plt.subplot(gs[0, 0])
    #query_swim = np.swapaxes(query_img,0,2)
    query_axs.set_title('Query Image')
    query_idx_m = query_idx if indexes is None else indexes[query_idx]
    query_axs.imshow(image_loader.get(query_idx_m))
    separator = np.zeros(shape=(320,5,3), dtype=np.uint8)

    for o_idx,o in enumerate(orders):
        _,ordered_dist,permut = o.get(query_idx, include_query=True, min_length=15000, keep_orig_consistency=True, cache_fld='DOP_cache_bis')
        n_permut = permut[:n]
        row = []
        for idx,p in enumerate(n_permut):
            img_idx = p if indexes is None else indexes[p]
            image = image_loader.get(img_idx)
            if idx == 0:
                row = image
                row = np.concatenate((row, separator), axis=1)
            else:
                row = np.concatenate((row, image, separator), axis=1)
        axs = plt.subplot(gs[o_idx+1, 0])
        axs.set_title(o.get_name(), loc='left')
        axs.set_yticklabels([])
        x = np.arange(240,480*n-230,480)
        labels = ['{:.5e}'.format(d) for d in ordered_dist[:n]]
        axs.set_xticks(x)
        axs.set_xticklabels(labels)
        axs.imshow(row)
    
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rankings visualization')
    parser.add_argument('--from-idx', type=int, default=0,
                        help='index of the image to use as query')
    parser.add_argument('--to-idx', type=int, default=5,
                        help='index of the last query image to include in results')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='enables features normalization')
    parser.add_argument('--clevr-dir', type=str, default='.',
                        help='CLEVR dataset base dir')
    parser.add_argument('--n', type=int, default=10,
                        help='number of images for every row')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='final image scale factor')
    parser.add_argument('--set', type=str, default='test', choices=['test','train'],
                        help='which set should be used')
    args = parser.parse_args()

    feats_dir = './features'
    feats_dir_already_filtered_objs = './features_clevr_lessequalthan5objs'

    # calculate indexes of images having less than MAX_OBJS
    set = 'train' if args.set == 'train' else 'val'
    cut = 15000 if args.set == 'train' else 2500
    scene_json_filename = os.path.join(args.clevr_dir, 'scenes', 'CLEVR_{}_scenes.json'.format(set))
    with open(scene_json_filename) as f:
        j = json.load(f)
    scenes = j['scenes'][:cut]
    scenes_idxs = [idx for idx, scene in enumerate(scenes) if len(scene['objects']) <= MAX_OBJS]
    print('Number of chosen images: {}'.format(len(scenes_idxs)))

    #initialize orders objects
    print('Initializing all orderings...')
    orders = []
    how_many = 15000
    #scene_json_filename = os.path.join(args.clevr_dir, 'scenes', 'CLEVR_val_scenes.json')
    orders.append(graphs_approx_order.GraphsApproxOrder(args.clevr_dir, 'proportional', how_many, args.set, indexes=scenes_idxs))
    orders.append(rmac_order.RMACOrder(os.path.join(feats_dir,'clevr_rmac_features.h5'),
        os.path.join(feats_dir,'clevr_rmac_features_order.txt'), args.normalize, how_many, args.set, indexes=scenes_idxs))
    
    #orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_avgconv_features_fp.pickle'), 'conv\navg fp', args.normalize))
    orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc2_avg_features_fp.pickle'), 'g_fc2\navg fp', args.normalize, indexes=scenes_idxs))
    #orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_512dim_bidir2_features_fp.pickle'), 'afteraggr\nbidir2 fp', args.normalize))
    orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_512dim_bidir2_transferlearn_weightedsum_epoch795_features_fp.pickle'), 'afteraggr\nbidir2\nTL-WS\nfp', args.normalize, indexes=scenes_idxs))
    orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_bs32_features_test.pickle'), 'G2V', args.normalize, indexes=scenes_idxs))
    orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_node_dropout_bs64_features_test.pickle'), 'G2V\nnode_dropout', args.normalize, indexes=scenes_idxs))
    orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_contrastive_mpnn1_selfatt_norm.pickle'), 'G2V\nself-att\nMPNN\nnorm', args.normalize, indexes=scenes_idxs))
    # note: the following call does not receive the indexes because the file already contains filtered images
    orders.append(rn_order.RNOrder(os.path.join(feats_dir_already_filtered_objs,'learning_to_rank_regression_957images_lessequalthan5objs_test.pickle'),
                                             'learn-to-rank\nregression', args.normalize, distance='cosine'))
    

    #build images
    img_dir = os.path.join(args.clevr_dir,'images')
    img_loader = ClevrImageLoader(img_dir, args.set)
    with PdfPages('images_out.pdf') as pdf:
        progress = trange(args.from_idx, args.to_idx+1)
        for idx in progress:
            fig = build_figure(orders, img_loader, idx, args.n, args.scale, scenes_idxs)
            pdf.savefig(fig)
