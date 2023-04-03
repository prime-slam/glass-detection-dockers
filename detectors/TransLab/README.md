# Segment_Transparent_Objects

[Original Implementation](https://github.com/xieenze/Segment_Transparent_Objects)

This repository contains the data and code for ECCV2020 paper [Segmenting Transparent Objects in the Wild](https://arxiv.org/abs/2003.13948).

For downloading the data, you can refer to [Trans10K Website](https://xieenze.github.io/projects/TransLAB/TransLAB.html).


## Installation
Copy the repository and navigate to this directory
```shell
git clone https://github.com/prime-slam/glass-detection-dockers.git
cd glass-detection-dockers/detectors/TransLab
```
Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide) for GPU support.
(You can omit this step if you intend to run the algorithm only on CPU)

Build the docker image
```shell
docker build -t translab -f Dockerfile ..
```


## Usage
### Predicting
- INPUT_DIR should contain target images.
- OUTPUT_DIR will store the generated masks and log.

Running the docker image:
```
docker run --rm --gpus all \
-v INPUT_DIR:/detector/input \
-v OUTPUT_DIR:/detector/output \
translab
```
### Predicting and Evaluating metrics.
- INPUT_DIR should contain target images.
- OUTPUT_DIR will store the generated masks, log and calculated metrics for each image.
- GT_DIR should contain ground truth grayscale/binary masks, each should have the **same name** as the corresponding image in INPUT_DIR.

Running the docker image:
```
docker run --rm --gpus all \
-v INPUT_DIR:/detector/input \
-v OUTPUT_DIR:/detector/output \
-v GT_DIR:/detector/ground_truth \
translab
```

To run the model on CPU instead of GPU, omit the `--gpus all` flag.


## License

For academic use, this project is licensed under the Apache License - see the LICENSE file for details. For commercial use, please contact the authors. 

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

```
@article{xie2020segmenting,
  title={Segmenting Transparent Objects in the Wild},
  author={Xie, Enze and Wang, Wenjia and Wang, Wenhai and Ding, Mingyu and Shen, Chunhua and Luo, Ping},
  journal={arXiv preprint arXiv:2003.13948},
  year={2020}
}
```
