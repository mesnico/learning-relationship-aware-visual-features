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
#Download RMAC features
echo "Downloading RMAC features..."

#Download cached GED-approx distances
echo "Downloading GED-approx precalculated distances"

#Extract features using RelationNetwork-CLEVR submodule
cd RelationNetworks-CLEVR
chmod u+x extract_features.sh
./extract_features.sh $CLEVRDIR
if [ -z "$(ls -A features)" ]; then
	echo "Something wrong while extracting features" >&2
	exit
fi

echo "Setup DONE"
