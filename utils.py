import pickle
import os
import numpy as np

def load_images(basedir):
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
    
    return elems
