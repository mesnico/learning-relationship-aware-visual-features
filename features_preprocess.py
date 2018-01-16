import numpy as np
import pickle

def load_features(filename):
    features = []

    f = open(filename, 'rb')
    while 1:
        try:
            this_feat = pickle.load(f)
            features.append(this_feat[1])
            bs = this_feat[1].shape[0]
            #print('batch #{} (size: {}) loaded'.format(this_feat[0], bs))
        except EOFError:
            break
    #pdb.set_trace()
    #print('features loaded from {}'.format(filename))
    return features

def process_fc_features(loaded_features):
    features = np.vstack(loaded_features)
    max_features = np.amax(features,axis=1)
    avg_features = np.mean(features,axis=1)

    print('processed #{} features each of size {}'.format(features.shape[0], features.shape[1]))
    return {'g_fc4_max':max_features, 'g_fc4_avg':avg_features}

def process_conv_features(loaded_features):
    global_max = []
    global_avg = []
    flatten = []
    subwindow3 = []
    for f in loaded_features:
        #calculate indexes over the entire image (5x5)
        global_max.append(np.amax(f,axis=(2,3)))
        global_avg.append(np.mean(f,axis=(2,3)))
        flatten.append(np.reshape(f,(f.shape[0],-1)))

        '''#calculate maximum over subwindows of size 3x3
        m = nn.MaxPool2d(3, stride=2)
        i = autograd.Variable(from_numpy(f))
        o = m(i)
        o = o.data.numpy()
        o = np.reshape(o,(o.shape[0],-1))'''
        #compute manually a max pooling with window size=3 and stride=2
        o1 = np.amax(f[:,:,0:3,0:3],axis=(2,3))
        o2 = np.amax(f[:,:,2:5,0:3],axis=(2,3))
        o3 = np.amax(f[:,:,0:3,2:5],axis=(2,3))
        o4 = np.amax(f[:,:,2:5,2:5],axis=(2,3))
        #take also the central subwindow
        o5 = np.amax(f[:,:,1:4,1:4],axis=(2,3))
        subwindow3.append(np.concatenate((o1,o2,o3,o4,o5),axis=1))

    global_max = np.vstack(global_max)
    global_avg = np.vstack(global_avg)
    flatten = np.vstack(flatten)
    subwindow3 = np.vstack(subwindow3)

    return ({'conv_max':global_max, 'conv_avg':global_avg, 'conv_flatten':flatten, 'conv_3x3_max':subwindow3})
        

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
