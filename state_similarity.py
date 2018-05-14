import itertools
import json
import pickle
import numpy as np
import pdb
from numpy.linalg import inv

'''DEBUG CODE'''
import matplotlib.pyplot as plt
from image_loader import ClevrImageLoader
images = ClevrImageLoader('../../../CLEVR_v1.0/images')
idx = 6

class StatesDistance:
    class IntersectablePair:
        def __init__(self, id_obj_1, id_obj_2, pair):
            self.id_obj_1 = id_obj_1
            self.id_obj_2 = id_obj_2
            self.pair = pair
        def __eq__(self, other):
            return self.id_obj_1, self.id_obj_2 == other.id_obj_1, other.id_obj_2
        def __hash__(self):
            return hash((self.id_obj_1,self.id_obj_2))

    def __init__(self, state_file):
        self.states = self.load_states(state_file)
        self.obj_dictionary = {}


    def load_states(self, state_file):
        cached_scenes = state_file.replace('.json', '.simil.pkl')

        '''if os.path.exists(cached_scenes):
            print('==> using cached scenes: {}'.format(cached_scenes))
            with open(cached_scenes, 'rb') as f:
                objects = pickle.load(f)
        else:'''
        objects = []
        with open(state_file, 'r') as json_file:
            scenes = json.load(json_file)['scenes']
            #print('caching all objects in all scenes...')
            for s in scenes:
                obj = s['objects']
                objects.append(obj)
            '''with open(cached_scenes, 'wb') as f:
                pickle.dump(all_scene_objs, f)'''

        return objects

    #returns unique ids for equal objects
    def get_object_id(self, obj_state):
        #attributes used as primary key for identifying an object
        keying_attributes = ['color','size','shape','material']
        # keys for accessing obj dictionary exclude 3d_coords
        key = {k:v for k,v in obj_state.items() if k in keying_attributes}
        key = str(key)
        if key not in self.obj_dictionary:
            self.obj_dictionary[key] = len(self.obj_dictionary)+1
        return self.obj_dictionary[key]

    #computes distance among relations (pairs) in two different scenes
    def distance_fn(self, scene_idx, scene1_dict, scene2_dict):
        #transforms position space into an orthonormal one (clevr does not use orthonormal basis)
        #TODO: get transform matrix components from scene json
        def clevr_transform(vector):
            basis_transform_matrix = np.matrix([[0.6563112735748291, -0.754490315914154],
                                                [0.7544902563095093, 0.6563112735748291]])
            return np.squeeze(np.asarray(np.matmul(inv(basis_transform_matrix),vector)))

        #calculate intersection between keys (permutation ids)
        intersection = set(scene1_dict.keys()).intersection(scene2_dict.keys())

        if len(intersection) == 0:
            #return maximum distance
            return 1.0
        
        #retrieve intersecting objects
        inters_objects_scene1 = [scene1_dict[k] for k in intersection]
        inters_objects_scene2 = [scene2_dict[k] for k in intersection]

        #calculate differences among pairs (we care about only x and y)
        deltapos_objs_scene1 = [clevr_transform(np.asarray(o1['3d_coords'][0:2]) - np.asarray(o2['3d_coords'][0:2])) for o1,o2 in inters_objects_scene1]
        deltapos_objs_scene2 = [clevr_transform(np.asarray(o1['3d_coords'][0:2]) - np.asarray(o2['3d_coords'][0:2])) for o1,o2 in inters_objects_scene2]
        
        dist = 0
        for dpos_obj_1, dpos_obj_2 in zip(deltapos_objs_scene1, deltapos_objs_scene2):
            signs = [(dpos_obj_1[i] > 0) == (dpos_obj_2[i] > 0) for i in range(2)]

            #distance chosen on the basis of coordinate signs match
            if signs[0] != signs[1]:          dist += 2/3
            elif signs[0] and signs[1]:         dist += 1/3
            elif not signs[0] and not signs[1]: dist += 1

        pdb.set_trace()

        ''' DEBUG CODE 
        print('n intersections: {}'.format(len(intersection)))
        for pair, o1, o2, deltapos1, deltapos2 in zip(intersection, inters_objects_scene1, inters_objects_scene2, deltapos_objs_scene1,deltapos_objs_scene2):
            print('positions : {}: {};{} and {};{}'.format(pair, o1[0]['3d_coords'][0:2],o1[1]['3d_coords'][0:2], o2[0]['3d_coords'][0:2],o2[1]['3d_coords'][0:2]))
            print('differences: {}: {} and {}'.format(pair, deltapos1, deltapos2))
            print('Dist: {}'.format(dist))
            plt.title('scene {}'.format(scene_idx))
            plt.imshow(images.get(scene_idx))
            plt.show()'''

        #normalization
        dist = dist / len(intersection) #2*dist / (len(scene1_dict) + len(scene2_dict))
        return dist

    def compute_distances(self, query_img_index):
        #get states of the query image
        query_scene = self.states[query_img_index]

        distances = []
        for scene_idx,curr_scene in enumerate(self.states):
            #create permutations
            #TODO: check for 'combinations' instead of permutations
            query_scene_permuts = list(itertools.combinations(query_scene, r=2))            
            curr_scene_permuts = list(itertools.combinations(curr_scene, r=2))

            #create permutations ids
            curr_scene_pairs = [(self.get_object_id(s[0]), self.get_object_id(s[1]))
                for s in curr_scene_permuts]
            query_scene_pairs = [(self.get_object_id(s[0]), self.get_object_id(s[1]))
                for s in query_scene_permuts]

            #create dictionaries from permutation ids to objects
            curr_scene_dict = {str(k):v for k,v in zip(curr_scene_pairs, curr_scene_permuts)}
            query_scene_dict = {str(k):v for k,v in zip(query_scene_pairs, query_scene_permuts)}
            
            d = self.distance_fn(scene_idx, curr_scene_dict, query_scene_dict)
            distances.append(d)
        return distances

#simple test
import os
if __name__ == "__main__":
    clevr_dir = '../../../CLEVR_v1.0'
    
    scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_val_scenes.json')
    s = StatesDistance(scene_json_filename)
    s.compute_distances(idx)
