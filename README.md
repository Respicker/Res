
Respicker
=============
## Installation
* Download repository
* Install dependencies: `pip install -r requirements.txt`


## Train models(just a small data to show the code really work)

python ./bin/resnet_train.py --tfrecords_dir data/train/  --checkpoint_dir model


## validate
python ./bin/resnet_eval.py --tfrecords_dir data/test --checkpoint_path output/unet_capital/  --batch_size 1000 --output_dir output/unet --events



### Detection From .mseed
python ./bin/resnet_eval_from_stream.py --stream_path ./mseed  --checkpoint_path model --batch_size 16 --output_dir ./data/out --plot True --save_sac False



more to come

