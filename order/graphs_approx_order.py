import pickle
import numpy as np
import pdb
import networkx as nx
from .AproximatedEditDistance import AproximatedEditDistance
import json
from .order_base import OrderBase
from .parallel_dist import parallel_distances

class ApproxGED(AproximatedEditDistance):
    """
        Vanilla Aproximated Edit distance, implements basic costs for substitution insertion and deletion.
    """
    def __init__(self, node_weight_mode='proportional'):
        self.node_weight_mode = node_weight_mode

    """
        Node edit operations
    """
    def node_substitution(self, g1, g2):
        def node_subst_cost_proportional(uattr, vattr):
            cost = 0
            attributes = list(uattr.keys())
            unit_cost = 1/len(attributes)
            for attr in attributes:
                if(uattr[attr] != vattr[attr]): cost=cost+unit_cost
            return cost

        def node_subst_cost_atleastone(uattr, vattr):
            cost = 0
            attributes = list(uattr.keys())
            for attr in attributes:
                if(uattr[attr] != vattr[attr]): 
                    cost=1
                    break
            return cost
        """
            Node substitution costs
            :param g1, g2: Graphs whose nodes are being substituted
            :return: Matrix with the substitution costs
        """
        if self.node_weight_mode == 'proportional':
            node_subst_cost = node_subst_cost_proportional
        elif self.node_weight_mode == 'atleastone':
            node_subst_cost = node_subst_cost_atleastone

        m = len(g1.nodes)
        n = len(g2.nodes)
        c_mat = np.array([node_subst_cost(g1.nodes[u], g2.nodes[v])
            for u in g1.nodes for v in g2.nodes]).reshape(m, n)
        return c_mat

    def node_insertion(self, g):
        """
            Node Insertion costs
            :param g: Graphs whose nodes are being inserted
            :return: List with the insertion costs
        """
        values = [v for k, v in g.nodes(data=True)]
        return [1]*len(values)

    def node_deletion(self, g):
        """
            Node Deletion costs
            :param g: Graphs whose nodes are being deleted
            :return: List with the deletion costs
        """
        values = [v for k, v in g.nodes(data=True)]
        return [1] * len(values)
'''
    """
        Edge edit operations
    """
    def edge_substitution(self, g1, g2):
        """
            Edge Substitution costs
            :param g1, g2: Adjacency list for particular nodes.
            :return: List of edge deletion costs
        """
        edge_dist = np.zeros((len(g1), len(g2)))
        return edge_dist

    def edge_insertion(self, g):
        """
            Edge insertion costs
            :param g: Adjacency list.
            :return: List of edge insertion costs
        """
        insert_edges = [len(e) for e in g]
        return np.ones(len(insert_edges))

    def edge_deletion(self, g):
        """
            Edge Deletion costs
            :param g: Adjacency list.
            :return: List of edge deletion costs
        """
        del_edges = [len(e) for e in g]
        return np.ones(len(del_edges))
'''
'''Ordering for distances extracted by graphs by means of approximated graph edit distance'''
class GraphsApproxOrder(OrderBase):
    graphs = None

    def __init__(self, gt='proportional', ncpu=4):
        super().__init__()
        if not GraphsApproxOrder.graphs:
            print('Building graphs from JSON...')
            GraphsApproxOrder.graphs = self.load_graphs('./')
        self.gt = gt
        self.ncpu = ncpu

    #Calculates approximated graph edit distance.
    def load_graphs(self,basedir):
        print('loading only the images...')
        dirs = os.path.join(basedir,'data')
        filename = os.path.join(dirs,'sort-of-clevr.pickle')
        with open(filename, 'rb') as f:
          train_datasets, test_datasets = pickle.load(f)

        elems = []
        for elem in train_datasets:
            img = elem[0]
            img = np.swapaxes(img,0,2)

            #Append also the graph is present in the data
            if len(elem)==3:
                elems.append((img))
            else:
                elems.append((img,elem[3]))

        for elem in test_datasets:
            img = elem[0]
            img = np.swapaxes(img,0,2)

            #Append also the graph is present in the data
            if len(elem)==3:
                elems.append((img))
            else:
                elems.append((img,elem[3]))
        print('loaded {} images'.format(len(elems)))
        #pdb.set_trace()
        graphs = [e[1] for e in elems]
        sep_graphs = [{'closest':g.copy(), 'farthest':g.copy()} for g in graphs]
        for g_set in sep_graphs:
            for k,g in g_set.items():
                if k=='closest':
                    rem_edges = [k for k,v in nx.get_edge_attributes(g,'relation').items() if v=='farthest']
                    g.remove_edges_from(rem_edges)
                elif k=='farthest':
                    rem_edges = [k for k,v in nx.get_edge_attributes(g,'relation').items() if v=='closest']
                    g.remove_edges_from(rem_edges)
        return sep_graphs


    def ged(self,g1,g2,node_weight_mode='proportional'):
        tot_cost = 0
        approx_ged = ApproxGED(node_weight_mode)
        for rel in ['closest','farthest']:
            c, _ = approx_ged.ged(g1[rel], g2[rel])
            tot_cost += c

        return tot_cost

    def compute_distances(self, query_img_index):
        return parallel_distances('ged-approx-{}'.format(self.gt), self.graphs, query_img_index, self.ged, kwargs={'node_weight_mode':self.gt}, ncpu=self.ncpu)
        #query_graph = self.graphs[query_img_index]
        #return [self.ged(query_graph, g) for g in self.graphs]

    def get_name(self):
        return 'graph GT\n({})\napprox'.format(self.gt)
    def length(self):
        return len(self.graphs)


#simple test
import os
if __name__ == "__main__":
    clevr_dir = '../../../../CLEVR_v1.0'
    idx = 6
    
    scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_val_scenes.json')
    s = GraphsApproxOrder(scene_json_filename)
    print(s.get(idx))
