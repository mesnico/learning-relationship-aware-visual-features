#!/bin/bash

set -e

get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

CLEVRDIR=$1
if [ -z "$CLEVRDIR" ]; then
	echo "Please specify CLEVR dataset base directory as first argument"
	exit
fi

if [ ! -d "$CLEVRDIR" ]; then
	echo "Specified CLEVR directory does not exist!"
	exit
fi

CLEVRDIR=$(get_abs_filename $CLEVRDIR)
echo "CLEVR is at "$CLEVRDIR
if [ ! -f .downloadok ]; then
	#Download RMAC features
	echo "Downloading RMAC features..."
	wget http://datone.isti.cnr.it/r-cbir/rmac_features.tar.xz

	#Download cached GED-approx distances
	echo "Downloading GED-approx precalculated distances"
	wget http://datone.isti.cnr.it/r-cbir/dist_cache.tar.xz

	echo "Extracting..."
	tar -xvf rmac_features.tar.xz
	tar -xvf dist_cache.tar.xz

	touch .downloadok
else
	echo "RMAC and GED distances already downloaded"
fi

#Extract features using RelationNetwork-CLEVR submodule
cd RelationNetworks-CLEVR
chmod u+x extract_features.sh
./extract_features.sh $CLEVRDIR
if [ -z "$(ls -A features)" ]; then
	echo "Something wrong while extracting features" >&2
	exit
fi

cd ..
echo "Building main virtual environment"
if [ ! -f .venvok ]; then
	mkdir retrieval_env
	virtualenv -p /usr/bin/python3 retrieval_env

	source ./retrieval_env/bin/activate

	echo "Installing dependencies..."
	which pip3
	pip3 install -r requirements.txt
	pip3 install -e networkx

	deactivate
	touch .venvok
else
	echo "Features virtual environment already installed"
fi

echo "Setup DONE"
