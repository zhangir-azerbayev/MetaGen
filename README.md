# MetaGen

## Demo

MetaGen takes as input the output of an object detection system. In this example, the object detection system is RetinaNet.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

MetaGen infers what objects are actually present.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

In doing so, MetaGen outperforms even RetinaNet with a confidence threshold fitted to maximize accuracy.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")


## Generating Data

The data used in the paper is provided in JSON format in `data_labelled`. Alternatively, this repo also provides code for procedurally generating data. 
First, load the Conda environment for data generation. 
```
$ conda create -f make_data/metagen_data_env.yaml 
$ conda activate metagen_data_env
```
Generate the unlabelled data (data without neural network detections) by running 
```
$ python data/gen_unlabelled.py
```
Next, set up one an object detector by running
```
$ python data/models/download_[model].py
```
where `[model]` is replaced by `retinanet`, `detr`, or `fasterrcnn`. Then generate the labelled data by running
```
$ python gen_labelled.py [model]
```
## Training MetaGen
To install the MetaGen Julia package, run the following Julia code. 
```
import Pkg
Pkg.develop("MetaGen/")
```
Hyperparameters for training MetaGen are set in a `yaml` file located in `configs`. To train MetaGen given `example_config.yaml`, run
```
julia MetaGen/scripts/train_metagen.jl configs/example_config.yaml
```

## Preprocessing the outputs from MetaGen
The outputs from running MetaGen (two .csv files, and one .json) are pre-processed in Julia using the scripts in the MetaGen/Preprocessing folder.

## Reproducing figures from the paper
The figures are generated using the R scripts in the Analysis folder. learning_V.R generates figure 1A. average_accuaracy_across_videos.R generates figures 3B and 3C.
