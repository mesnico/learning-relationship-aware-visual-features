import json
import pickle
import os
import pdb
import re
from tqdm import tqdm

feat_name = 'g_fc2_avg-fp-normTrue-settrain-len15000'
cache_dir = os.path.join('DOP_cache',feat_name)
out_folder = 'static_rankings_json'
digits = 6

if not os.path.exists(out_folder):
	os.makedirs(out_folder)

for filename in tqdm(os.listdir(cache_dir)):
	json_root = {}
	match = re.search(r'dop-(\d+).pkl', filename)
	query_id = match.group(1)

	with open(os.path.join(cache_dir,filename),'rb') as f:
		_, ordered_distances, permuts = pickle.load(f)

	od = ordered_distances.tolist()
	#round floats to 6 digits
	od = [round(x,digits) for x in od]
	json_root['ordered_distances'] 	= od
	json_root['permuts'] 			= permuts.tolist()


	with open(os.path.join(out_folder, '{}_{}.json'.format(feat_name, query_id)), 'w') as j:
		json.dump(json_root, j)
