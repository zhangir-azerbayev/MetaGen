# MetaGen

## Generating Data

The data used in the paper is provided in JSON format in `data_labelled`. Alternatively, this repo also provides code for procedurally generating data. 
First, load the Conda environment for data generation. 
```
$ conda create -f make_data/metagen_data_env.yaml 
$ conda activate metagen_data_env
```
Generate the unlabelled data (data without neural network detections) by running 
```
$ python make_data/gen_unlabelled.py
```
Next, set up one an object detector by running
```
$ python make_data/models/download_[model].py
```
where `[model]` is replaced by `retinanet`, `detr`, or `fasterrcnn`. Then generate the labelled data by running
```
$ python gen_labelled.py [model]
```
