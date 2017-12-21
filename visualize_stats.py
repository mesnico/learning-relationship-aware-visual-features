import numpy as np
import argparse
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pdb
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Stats visualizer')
parser.add_argument('--ground-truth', type=str, choices=['proportional','atleastone'], default='proportional',
                    help='how many images in the result')
parser.add_argument('--aggregate', action='store_true', default=False,
                    help='enable max aggregation on multiple stats files')
parser.add_argument('--confidence', type=float, default=0.95,
                    help='confidence interval')
args = parser.parse_args()

stats_dir = './stats'
merged_stats = {}
filenames = {}
filenames['normalized'] = os.path.join(stats_dir, 'stats_normalized_{}-gt.pickle'.format(args.ground_truth))
filenames['no-normalized'] = os.path.join(stats_dir, 'stats_no-normalized_{}-gt.pickle'.format(args.ground_truth))
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

def build_bar_graph(merged_stats, name, max_grouping=False, confidence=0.95):
    width=0.27
    error_formatting = dict(elinewidth=1, capsize=2)
    i = 0
    ind = 0
    bars = []
    fig, ax = plt.subplots()
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
        for feat in feats:
            values = [e[feat] for e in stat]
            #pdb.set_trace()
            mean = np.mean(values)
            mean_values.append(mean)
            sem = st.sem(values)
            conf = st.t.ppf((1+confidence)/2, len(values)-1) * sem
            y_errors.append(conf)
        all_means.append(mean_values)
        all_y_errors.append(y_errors)
        num_bars = len(feats)
        ind = np.arange(1, num_bars+1)
        if not max_grouping:
            bars.append(plt.bar(ind+width*i, mean_values, width, color=colors[i], yerr=np.array(y_errors), error_kw=error_formatting))
        i=i+1
        ax.set_xticks(ind)
        ax.set_xticklabels(feats.keys())
    
    if max_grouping:
        #calculate the best mean for every feat
        max_mean = np.amax(np.asarray(all_means),0)
        std_indexes = np.argmax(np.asarray(all_means),0)
        
        chosen_y_error = [all_y_errors[row][col] for col,row in enumerate(std_indexes)]
        plt.bar(ind, max_mean, width, color=colors[0], yerr=np.array(chosen_y_error), error_kw=error_formatting)
    else:
        ax.legend(bars, list(merged_stats.keys()) )
    ax.set_ylabel('{} index'.format(name))
    ax.set_title('{}, {}% conf. interval'.format(name,confidence*100))

def build_recall_graph(merged_stats, confidence = 0.95):
    view = {}
    for typekey, ftype in merged_stats.items():
        #typekes at this moment can be only 'normalized' or 'no-normalized'
        stat = ftype['recall-at-k']
        #eliminate columns i'm not interested in
        feats = {k:stat[0][k] for k in stat[0] if k not in 'image_id' and k not in 'n_items'}
        
        #reorganize the values so that i have all the samples for every feat
        for sample in stat:
            for feat,v in sample.items():
                if feat not in feats:
                    continue
                s = []
                #line_legend = '{} - {}'.format(feat, typekey)
                if typekey not in view:
                    view[typekey] = {}
                if feat not in view[typekey]:
                    view[typekey][feat] = []
                for k,recall in OrderedDict(sorted(v.items())).items():
                    s.append((k,recall))
                view[typekey][feat].append(s)

    #calculate mean and conf interval for every point
    fig, ax = plt.subplots(len(view))
    i = 0
    for typekey, ftype in view.items():
        if len(view) == 1:
            this_ax = ax
        else:
            this_ax = ax[i]

        lines = []
        for feat,v in ftype.items():
            a = np.asarray(v)
            recalls = a[:,:,1]
            k = a[0,:,0]
            recalls_mean = np.mean(recalls,axis=0)
            recalls_sem = st.sem(recalls,axis=0)
            conf = st.t.ppf((1+confidence)/2, a.shape[0]-1) * recalls_sem
            
            lines.append(this_ax.errorbar(k, recalls_mean, yerr=conf, fmt='-', elinewidth=1, capsize=1))
            this_ax.set_xscale("log", nonposx='clip')
        this_ax.set_title('{}, {}% conf. interval'.format(typekey,confidence*100))
        this_ax.set_xlabel('k')
        this_ax.set_ylabel('recall-at-k index')
        this_ax.legend(lines, list(ftype.keys()))
        i=i+1

#display the graph for every statistic different from the recall-at-k
stats = list(list(merged_stats.values())[0].keys())
bar_stats = [s for s in stats if s not in 'recall-at-k']
for s in bar_stats:
    build_bar_graph(merged_stats, s, args.aggregate, args.confidence)

#display the grapg for the recall-at-k
build_recall_graph(merged_stats, args.confidence)
plt.show()
    




