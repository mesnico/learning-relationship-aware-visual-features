import matplotlib
matplotlib.use('Agg')    
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import cv2
import os
import argparse
import numpy as np
from order import rn_order, rmac_order, graphs_order, states_order, graphs_approx_order
from utils import load_images
from tqdm import tqdm, trange
import math

class SoCImageLoader():
    def __init__(self, images):
        self.images = images

    def get(self,index):
        im = np.swapaxes(self.images[index],0,2)
        return im

def build_figure(orders, image_loader, query_idx, n=10, scale=1):
    size = (math.ceil(4*n*scale), math.ceil(3*(len(orders)+1)*scale)+3)

    fig = plt.figure('Query idx {}'.format(query_idx), figsize=size)
    gs = gridspec.GridSpec(len(orders)+1, 1)

    #query_img
    query_axs = plt.subplot(gs[0, 0])
    #query_swim = np.swapaxes(query_img,0,2)
    query_axs.set_title('Query Image')
    query_axs.imshow(image_loader.get(query_idx))
    x_dim = image_loader.get(query_idx).shape[0]
    separator = np.zeros(shape=(x_dim,2,3))

    tmp_dic = {'graph GT\n(proportional)\napprox':'Ground-truth', 'g_fc4\nmax':'Two-stage RN', 'RMAC':'RMAC'}

    for o_idx,o in enumerate(orders):
        _,ordered_dist,permut = o.get(query_idx, False)
        print('{} Query is in position {} and has distance {}'.format(o.get_name(), list(permut).index(query_idx), ordered_dist[list(permut).index(query_idx)]))
        n_permut = permut[:n]
        row = []
        n_permut_v = [v + 1 for v in n_permut]
        for idx,p in enumerate(n_permut_v):
            image = image_loader.get(p)
            if idx == 0:
                row = image
                row = np.concatenate((row, separator), axis=1)
            else:
                row = np.concatenate((row, image, separator), axis=1)
        axs = plt.subplot(gs[o_idx+1, 0])
        axs.set_title(tmp_dic[o.get_name()], loc='left')
        axs.set_yticklabels([])
        x = np.arange(x_dim/2,x_dim*n-x_dim/2,x_dim)
        labels = ['{:.5e}'.format(d) for d in ordered_dist[:n]]
        #axs.set_xticks(x)
        #axs.set_xticklabels(labels)
        axs.set_xticks([])
        axs.set_xticklabels([])
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
    args = parser.parse_args()

    feats_dir = './features'  

    #initialize orders objects
    print('Initializing all orderings...')
    feats_orders = []
    #feats_orders.append(graphs_order.GraphsOrder('proportional', 4))
    feats_orders.append(graphs_approx_order.GraphsApproxOrder('proportional', 4))
    #feats_orders.append(graphs_order.GraphsOrder('atleastone', 4))
    #feats_orders.append(graphs_approx_order.GraphsApproxOrder('atleastone', 4))

    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'avg_features_sd.pickle'), 'g_fc2\navg sd', args.normalize))
    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'avg_features_fc.pkl'), 'g_fc4\navg', args.normalize))
    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'max_features_sd.pickle'), 'g_fc2\nmax sd', args.normalize))
    feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'max_features_fc.pkl'), 'g_fc4\nmax', args.normalize))
    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'avg_features_fc_prenorm.pkl'), 'g_fc4\navg\nprenorm', args.normalize))
    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'max_features_fc_prenorm.pkl'), 'g_fc4\nmax\nprenorm', args.normalize))
    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'max_features_conv.pkl'), 'conv\nmax', args.normalize))
    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'avg_features_conv.pkl'), 'conv\navg', args.normalize))
    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'afteraggr_features_sd.pickle'), 'afteraggr\nsd', args.normalize))
    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'afteraggr-no-prenorm_features_sd.pickle'), 'afteraggr\nsd\nno-prenorm', args.normalize))
    
    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc0_avg_features_original_fp.pickle'), 'conv\navg fp\noriginal', args.normalize))
    #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc0_max_features_original_fp.pickle'), 'conv\nmax fp\noriginal', args.normalize))
    feats_orders.append(rmac_order.RMACOrder(os.path.join(feats_dir,'sort-of-clevr_S50+75+100.h5'), args.normalize))

    #build images
    elems = load_images('./')
    images = [e[0] for e in elems]
    img_loader = SoCImageLoader(images)
    with PdfPages('images_out.pdf') as pdf:
        progress = trange(args.from_idx, args.to_idx+1)
        for idx in progress:
            fig = build_figure(feats_orders, img_loader, idx, args.n, args.scale)
            plt.savefig('img_out.png')
            pdf.savefig(fig)    
    
            
        

    
