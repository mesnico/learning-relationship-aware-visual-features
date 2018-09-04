import numpy as np
import argparse
import pickle
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pdb
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Stats visualizer')
parser.add_argument('--ground-truth', type=str, default='graph-approx-proportional',
                    help='how many images in the result')
parser.add_argument('--aggregate', action='store_true', default=False,
                    help='enable max aggregation on multiple stats files')
parser.add_argument('--confidence', type=float, default=0.95,
                    help='confidence interval')
parser.add_argument('--scale', type=float, default=2.0,
                    help='graphs scale factor')
parser.add_argument('--set', type=str, default='test', choices=['test','train'],
                    help='which set should be used')
args = parser.parse_args()

stats_dir = './stats'
merged_stats = {}
filenames = {}
filenames['normalized'] = os.path.join(stats_dir, 'stats_normalized_{}-gt_{}.pickle'.format(args.ground_truth, args.set))
filenames['no-normalized'] = os.path.join(stats_dir, 'stats_no-normalized_{}-gt_{}.pickle'.format(args.ground_truth, args.set))
for stat_type, filename in filenames.items():
    if os.path.isfile(filename):
        f = open(filename,'rb')
        merged_stats[stat_type] = pickle.load(f)

#if no valid file is found, raise an error
if len(merged_stats) == 0:
    raise IOError('No valid stats files found!')

'''
Builds a bar graph for every stat
'stat' is a list of dictionary entries, one for every feature. Every element of the list is a different sample
'name' is the name of that stat
'''

colors = ['r','g','b']

def build_bar_graph(merged_stats, name, max_grouping=False, can_be_negative=True, confidence=0.95, scale=1.0):
    width=0.27
    error_formatting = dict(elinewidth=1, capsize=2)
    i = 0
    ind = 0
    feat_sorted_keys = []
    bars = []
    fig, ax = plt.subplots(figsize=(5*scale, 4*scale))
    plt.gcf().subplots_adjust(left=0.25)
    all_means = []
    all_y_errors = []
    for typekey, ftype in merged_stats.items():
        #typekes at this moment can be only 'normalized' or 'no-normalized'
        stat = ftype[name]
        #calculate mean values and confidence intervals
        mean_values = []
        y_errors = []
        #eliminate columns i'm not interested in
        feats = {k:stat[0][k] for k in stat[0] if k not in 'image_id' and k not in 'n_items'}
        feats_sorted_keys = sorted(feats)
        for feat in feats_sorted_keys:
            values = [e[feat] for e in stat]
            #pdb.set_trace()
            mean = np.mean(values)
            mean_values.append(mean)
            sem = st.sem(values)
            conf = st.t.ppf((1+confidence)/2, len(values)-1) * sem
            neg_conf = conf if mean>conf or can_be_negative else mean
            y_errors.append([neg_conf,conf])
            
            #f = feat.replace('\n',' ')
            #print('{} {} -  = {} +- {}'.format(f, typekey, mean, conf))
        all_means.append(mean_values)
        all_y_errors.append(y_errors)
        num_bars = len(feats)
        ind = np.arange(1, num_bars+1)
        if not max_grouping:
            bars.append(plt.bar(ind+width*i, mean_values, width, color=colors[i], yerr=np.transpose(np.array(y_errors)), error_kw=error_formatting))
        i=i+1
        ax.set_xticks(ind)
        ax.set_xticklabels(feats_sorted_keys)
    
    if max_grouping:
        #calculate the best mean for every feat
        max_mean = np.amax(np.asarray(all_means),0)
        std_indexes = np.argmax(np.asarray(all_means),0)
        
        chosen_y_error = [all_y_errors[row][col] for col,row in enumerate(std_indexes)]
        shading = [colors[1] if 'g_fc' in f else colors[2] for f in feats_sorted_keys] 
        yerr=np.transpose(np.array(chosen_y_error))
        plt.bar(ind, max_mean, width, color=shading, yerr=yerr, error_kw=error_formatting)
        
        for ind, k in enumerate(feats_sorted_keys):
            #pdb.set_trace
            f = k.replace('\n',' ')
            print('{}--{}: {:.2} +/-{:.2}'.format(name,f,max_mean[ind], yerr[0,ind]))

    else:
        ax.legend(bars, list(merged_stats.keys()) )
    ax.set_ylabel('{} index'.format(name))
    #ax.set_xlim(0, len(ind))
    ax.set_title('{}, {}% conf. interval'.format(name,confidence*100))
    ax.grid(color='r', linestyle='dotted', linewidth=1)

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#display the graph for every statistic different from the recall-at-k
with PdfPages(os.path.join(output_dir,'stats_out_{}-gt.pdf'.format(args.ground_truth))) as pdf:
    stats = list(list(merged_stats.values())[0].keys())
    bar_stats = [s for s in stats]
    for s in bar_stats:
        negative = True if s == 'spearmanr' else False
        build_bar_graph(merged_stats, s, args.aggregate, negative, args.confidence, args.scale)
        pdf.savefig()
    




