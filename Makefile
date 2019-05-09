all: /usr/bin/ffprobe ./checkpoint/d-pt-243.bin ./data/h36m.zip /home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/opencv_python*

/home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/opencv_python*:
	pip install opencv-python

/usr/bin/ffprobe:
	sudo apt-get install ffmpeg

./checkpoint/d-pt-243.bin:
	mkdir -p ./checkpoint && cp ../datasets/d-pt-243.bin ./checkpoint/

./data/h36m.zip:
	cd ./data && cp ../../datasets/h36m.zip . && python prepare_data_h36m.py --from-archive h36m.zip

run.video:
	python run_wild.py -k detections -arc 3,3,3,3,3 -c checkpoint --evaluate d-pt-243.bin --render --viz-subject S1 --viz-action Directions --viz-video InTheWildData/out_cutted.mp4 --viz-camera 0 --viz-output output_scater.mp4 --viz-size 5 --viz-downsample 1 --viz-skip 9

run:
	python run_wild2.py -arc 3,3,3,3,3 -c checkpoint --evaluate d-pt-243.bin --input ./data/data_2d_detections_scater.npz --output ./out_3D_vp3d.npz

2d:
	python npz2json2d.py ./data/data_2d_detections_scater.npz > ../results/skater2d.json

3d:
	python npz2json3d.py ./out_3D_vp3d.npz > ../results/output.json

.PHONY: run input
