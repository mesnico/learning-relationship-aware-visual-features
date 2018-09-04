#!/bin/bash

set -e

OPTIND=1

begin=249
end=258
CLEVRDIR=""

usage="\nUsage: $(basename "$0") -d clevr_dir [OPTIONS]\nOPTIONS:\n  -s\tSpecifies start query index [default: $begin]\n  -e\tSpecifies end query index [default: $end]\n"

while getopts "s:e:d:h" opt; do
    case "$opt" in
    s)  begin=$OPTARG
        ;;
    e)  end=$OPTARG
        ;;
    d)  CLEVRDIR=$OPTARG
	;;
    h)  printf "$usage"
        exit
        ;;
    esac
done

if [ -z "$CLEVRDIR" ]; then
	echo "CLEVR directory not specified. Please use -d option"
	exit
fi

if [ ! -d "$CLEVRDIR" ]; then
	echo "Specified CLEVR directory does not exist!"
	exit
fi

if [ ! -f .venvok ]; then
	echo "Virtual environment not installed. Have you run setup.sh?"
	exit
fi

source ./retrieval_env/bin/activate

python3 visualize_images.py --clevr-dir $CLEVRDIR --from-idx $begin --to-idx $end

deactivate
