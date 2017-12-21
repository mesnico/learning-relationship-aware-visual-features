#!/bin/bash

CLEVRDIR=$1

function run_docker() {
	docker run --rm -it --ipc=host \
		--volume=$PWD:/app \
		--volume=$CLEVRDIR:/clevr \
		-e "DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		clevr-ir python3 /app/main.py --query-img-index 0 --until-img-index 100 --clevr-dir /clevr --ground-truth $1 --cpus 50
}

if [ -z "$CLEVRDIR" ]; then
	printf "Please specify CLEVR dataset base directory as first argument\n"
else
	run_docker "proportional"
	run_docker "atleastone"
fi
