import argparse
import os
import json
import pdb
import networkx as nx
import similarity_search
import features_preprocess as fp
from image_loader import ClevrImageLoader
'''import importlib.util
spec = importlib.util.spec_from_file_location("similarity_search", "../../similarity_search_engine/similarity_search.py")
similarity_search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(similarity_search)'''

parser = argparse.ArgumentParser(description='Similarity search and stats recording script')
parser.add_argument('--query-img-index', type=int, default=10,
                    help='index of the image to use as query')
parser.add_argument('--until-img-index', type=int, default=None,
                    help='index of the last image to use as query (for stats recording)')
parser.add_argument('--cut', type=int, default=-1,
                    help='how many features of the vector are considered')
parser.add_argument('--N', type=int, default=10,
                    help='how many images in the result')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='enables features normalization')
parser.add_argument('--ground-truth', type=str, choices=['proportional','atleastone'], default='proportional',
                    help='how many images in the result')
parser.add_argument('--clevr-dir', type=str, default='.',
                    help='CLEVR dataset base dir')
parser.add_argument('--cpus', type=int, default=8,
                    help='how many CPUs to use for graph distance calculation')
parser.add_argument('--skip-missing', action='store_true', default=False,
                    help='skip missing cached distances')
parser.add_argument('--include-query', action='store_true', default=False,
                    help='include the query in the results')
args = parser.parse_args()


features_dirs = './features'

def load_graphs(clevr_scenes):
    graphs = []

    for scene in clevr_scenes:
        graph = nx.MultiDiGraph()
        #build graph nodes for every object
        objs = scene['objects']
        for idx, obj in enumerate(objs):
            graph.add_node(idx, color=obj['color'], shape=obj['shape'], material=obj['material'], size=obj['size'])
        
        relationships = scene['relationships']
        for name, rel in relationships.items():
            if name in ('right','front'):
                for b_idx, row in enumerate(rel):
                    for a_idx in row:
                        graph.add_edge(a_idx, b_idx, relation=name)

        graphs.append(graph)
    return graphs

images_dir = os.path.join(args.clevr_dir, 'images')

json_filename = os.path.join(args.clevr_dir, 'scenes', 'CLEVR_val_scenes.json')
clevr_scenes = json.load(open(json_filename))['scenes']
graphs = load_graphs(clevr_scenes)

#load and process rmac features
feat_filename = os.path.join(features_dirs,'clevr_rmac_features.h5')
feat_order_filename = os.path.join(features_dirs,'clevr_rmac_features_order.txt')
rmac_features = fp.load_rmac_features(feat_filename, feat_order_filename)

#load and process g_fc4 features
filename = os.path.join(features_dirs,'avg_features.pickle')
avg_fc4_features = fp.load_features('g_fc4_avg',filename)
filename = os.path.join(features_dirs,'max_features.pickle')
max_fc4_features = fp.load_features('g_fc4_max',filename)

#merge fc and conv features
features = {**avg_fc4_features, **max_fc4_features, **rmac_features}

if args.cut != -1:
    features = np.asarray([features[i][0:args.cut] for i in range(len(features))])

#print('Features have now shape {}'.format(features.shape))

if args.normalize :
    features = {name:(fp.normalized(feat[0], 1), feat[1]) for name, feat in features.items()}

#start
images_loader = ClevrImageLoader(images_dir)
similarity_search.start(images_loader, args, graphs, features)
