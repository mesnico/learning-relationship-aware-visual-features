import numpy
import pickle
import os

size = 512
num = 15000
outdir = 'features'

a = numpy.random.rand(num, size)
with open(os.path.join(outdir,'random.pickle'), 'wb') as f:
	pickle.dump([[0,a]], f)
