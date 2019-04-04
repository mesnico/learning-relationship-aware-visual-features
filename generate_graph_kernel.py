from order.graphs_approx_order import ApproxGED
from order.order_base import OrderBase
from order.parallel_dist import parallel_distances
import networkx as nx
import json
import os
import tqdm
import torch
import numpy as np

MAX_OBJS = 5


class GraphsApproxOrder(OrderBase):
    graphs = None

    def __init__(self, clevr_dir, gt='proportional', how_many=15000, st='test', ncpu=4):
        super().__init__()

        s = 'val' if st == 'test' else st
        scene_file = os.path.join(clevr_dir, 'scenes', 'CLEVR_{}_scenes.json'.format(s))
        if not GraphsApproxOrder.graphs:
            print('Building graphs from JSON from {}...'.format(s))
            GraphsApproxOrder.graphs, self.idxs = self.load_graphs(scene_file, how_many)
        self.gt = gt
        self.st = st
        self.ncpu = ncpu

    def load_graphs(self, scene_file, how_many):
        clevr_scenes = json.load(open(scene_file))['scenes']
        clevr_scenes = clevr_scenes[:how_many]
        graphs = []
        img_indexes = []

        for idx, scene in enumerate(clevr_scenes):
            graph = nx.MultiDiGraph()
            # build graph nodes for every object
            objs = scene['objects']
            if len(objs) > MAX_OBJS:
                continue
            for idx, obj in enumerate(objs):
                graph.add_node(idx, color=obj['color'], shape=obj['shape'], material=obj['material'], size=obj['size'])

            relationships = scene['relationships']
            for name, rel in relationships.items():
                if name in ('right', 'front'):
                    for b_idx, row in enumerate(rel):
                        for a_idx in row:
                            graph.add_edge(a_idx, b_idx, relation=name)
            img_indexes.append(idx)
            graphs.append(graph)
        return graphs, img_indexes

    '''
    Calculates approximated graph edit distance.
    '''

    def ged(self, g1, g2, node_weight_mode='proportional'):
        tot_cost = 0
        approx_ged = ApproxGED(self.gt)
        '''for rel in ['right','front']:
            c, _ = approx_ged.ged(g1[rel], g2[rel])
            tot_cost += c
        '''
        tot_cost, _ = approx_ged.ged(g1, g2)
        return tot_cost

    def compute_distances(self, query_img_index):
        return parallel_distances('ged-approx-{}-{}'.format(self.gt, self.st), self.graphs, query_img_index, self.ged,
                                  kwargs={'node_weight_mode': self.gt}, ncpu=self.ncpu)
        #query_graph = self.graphs[query_img_index]
        #return [self.ged(query_graph, g) for g in self.graphs]

    def get_name(self):
        return 'graph GT\n({})\napprox'.format(self.gt)

    def get_identifier(self):
        return '{}-set{}'.format(self.get_name().replace('\n', '_').replace(' ', '-'), self.st)

    def length(self):
        return len(self.graphs)


if __name__ == '__main__':
    ncpu = 14
    clevr_dir = '/home/nicola/Documents/CLEVR_v1.0'
    cache_fld = 'KernelDistances_cache'
    kernel = []
    st = 'test'
    how_many = 2500 if st != 'train' else 15000

    graph_order = GraphsApproxOrder(clevr_dir, st=st, how_many=how_many, ncpu=ncpu)
    print('Number of graphs having #OBJS<={}: {}. Taken: {}'.format(MAX_OBJS, graph_order.length(), how_many))
    for idx in tqdm.trange(graph_order.length()):
        distances, _, _ = graph_order.get(idx, min_length=15000, include_query=True, cache_fld=cache_fld)
        kernel.append(distances)

    np_kernel = np.array(kernel)
    max_every_row = np.amax(np_kernel, axis=1)
    max_every_row = np.expand_dims(max_every_row, axis=1)
    max_matrix = np.repeat(max_every_row, np_kernel.shape[1], axis=1)
    similarities = max_matrix - np_kernel
    similarities = torch.from_numpy(similarities)
    torch.save(similarities, os.path.join('ged_kernel_clevr_{}.dat'.format(st)))


