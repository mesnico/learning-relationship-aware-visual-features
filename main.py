import argparse
import os
import json
import pdb
import cv2
import networkx as nx
import similarity_search
import features_preprocess as fp
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
args = parser.parse_args()


features_dirs = './features'


class SoCImageLoader():
    def __init__(self, images_dir):
        self.images_dir = images_dir

    def get(self,index):
        padded_index = str(index).rjust(6,'0')
        img_filename = os.path.join(self.images_dir, 'val', 'CLEVR_val_{}.png'.format(padded_index))
        image = cv2.imread(img_filename)
        return image / 255.

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

feat_filename = os.path.join(features_dirs,'clevr_rmac_features.h5')
feat_order_filename = os.path.join(features_dirs,'clevr_rmac_features_order.txt')
rmac_features = fp.load_rmac_features(feat_filename, feat_order_filename)
rmac_features = fp.process_rmac_features(rmac_features)

#merge fc and conv features
features = rmac_features#= {**fc_features, **conv_features}

if args.cut != -1:
    features = np.asarray([features[i][0:args.cut] for i in range(len(features))])

#print('Features have now shape {}'.format(features.shape))

if args.normalize :
    features = {name:similarity_search.normalized(feat, 1) for name, feat in features.items()}

#start
images_loader = SoCImageLoader(images_dir)
similarity_search.start(images_loader, args.query_img_index, graphs, args.N, args.ground_truth, args.until_img_index, args.cpus, features)
