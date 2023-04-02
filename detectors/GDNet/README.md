# CVPR2020_GDNet

Original Repository: 

## Don't Hit Me! Glass Detection in Real-world Scenes
[Haiyang Mei](https://mhaiyang.github.io/), Xin Yang, Yang Wang, Yuanyuan Liu, Shengfeng He, Qiang Zhang, Xiaopeng Wei, and Rynson W.H. Lau

[[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Mei_Dont_Hit_Me_Glass_Detection_in_Real-World_Scenes_CVPR_2020_paper.pdf)] [[Project Page](https://mhaiyang.github.io/CVPR2020_GDNet/index.html)]

### Abstract
Glass is very common in our daily life. Existing computer vision systems neglect the glass and thus might lead to severe consequence, \eg, the robot might crash into the glass wall. However, sensing the presence of the glass is not straightforward. The key challenge is that arbitrary objects/scenes can appear behind the glass and the content presented in the glass region typically similar to those outside of it. In this paper, we raise an interesting but important problem of detecting glass from a single RGB image. To address this problem, we construct a large-scale glass detection dataset (GDD) and design a glass detection network, called GDNet, by learning abundant contextual features from a global perspective with a novel large-field contextual feature integration module. Extensive experiments demonstrate the proposed method achieves superior glass detection results on our GDD test set. Particularly, we outperform state-of-the-art methods that fine-tuned for glass detection.

### Citation
If you use this code or our dataset (including test set), please cite:

```
@InProceedings{Mei_2020_CVPR,
    author = {Mei, Haiyang and Yang, Xin and Wang, Yang and Liu, Yuanyuan and He, Shengfeng and Zhang, Qiang and Wei, Xiaopeng and Lau, Rynson W.H.},
    title = {Don't Hit Me! Glass Detection in Real-World Scenes},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

## Installation
Clone this repository:
```
git clone https://github.com/prime-slam/glass-detection-dockers.git
cd glass-detection-dockers/detectors/GDNet
```

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide) for GPU support.
(You can omit this step if you intend to run the algorithm only on CPU)

Then you need to build a docker image:
```
docker build -t gdnet -f Dockerfile ..
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
gdnet
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
gdnet
```

To run the model on CPU instead of GPU, omit the `--gpus all` flag.


## Miscellaneous
### Dataset
See [Peoject Page](https://mhaiyang.github.io/CVPR2020_GDNet/index.html)


### Test
If you intend to run the project without docker, download trained model `GDNet.pth` at [here](https://mhaiyang.github.io/CVPR2020_GDNet/index.html), then run `infer.py`.

### License
Please see `license.txt`

### Contact
Original Repository Author E-Mail: mhy666@mail.dlut.edu.cn \
Current Repository Author E-Mail: kiselev.0353@gmail.com
