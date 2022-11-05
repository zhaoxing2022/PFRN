<!-- TOC -->
* [Position Fusing and Refining for Clear Salient Object Detection](#position-fusing-and-refining-for-clear-salient-object-detection)
  * [Requirements](#requirements)
  * [Train](#train)
  * [Test](#test)
  * [Metrics](#metrics)
    * [Python](#python)
    * [Matlab](#matlab)
  * [Thanks](#thanks)
  * [Cite](#cite)
<!-- TOC -->
# [Position Fusing and Refining for Clear Salient Object Detection](https://ieeexplore.ieee.org/document/9940193)
## Requirements
Packages that not in Anaconda:
* Pytorch(1.12.1)/Torchvision(0.13.1)
* Opencv-python(4.6.0)

The code is organized from the draft, if you find any bugs or problems, you can contact me, thanks :)
## Train
We use [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) to train the model on multiple GPUs, it has better performance and faster than DataParallel. You can implement your own ideas in this framework.
1. Align the dataset folder directory with **[dataset/train](https://github.com/zhaoxing2022/PFRN/tree/main/dataset/train)**. 
2. Modify the parameters at the top of **[train/train.py](https://github.com/zhaoxing2022/PFRN/blob/main/train/train.py)**, I have made detailed annotation, you need to change it to your own configuration.
3. ```CUDA_VISIBLE_DEVICES=xxxx python train.py```. 
## Test
1. Download the [weights](https://drive.google.com/file/d/1GmPJPypGjcIZxSWhG6mEjcygKlb6mzpL/view?usp=sharing).  You can download the pre-calculated saliency maps of our method at [Google Drive](https://drive.google.com/file/d/1i92D5by5YtVySh35kJeVxyZLtXvbxYZ3/view?usp=sharing).
2. Align the dataset folder directory with **[dataset/test](https://github.com/zhaoxing2022/PFRN/tree/main/dataset/test)**. Because we don't read the labels during the testing stage, you can test any images.
3. Modify the parameters at the top of **[test/test.py](https://github.com/zhaoxing2022/PFRN/blob/main/test/test.py)**, I have made detailed annotation, you need to change it to your own configuration.
4. ```CUDA_VISIBLE_DEVICES=xxxx python test.py```. Our model supports multiple GPUs and batch size > 1 during inference. It is very fast.

## Metrics
Two types of metric calculations are provided.
### Python
- Modified from [SOCToolbox](https://github.com/mczhuge/SOCToolbox/tree/main/codes). The evaluation code uses multiple processes, which is much faster than the matlab version. There are slight differences with the matlab version, which can be used for quick evaluation. You can change your configuration at the bottom of [metrics.py](https://github.com/zhaoxing2022/PFRN/tree/main/metrics/python/eval_sod.py).
### Matlab
- A commonly used evaluation toolbox.

## Thanks
* [PR anf F-measure curves](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool).
* [Evaluate toolbox](https://github.com/weijun88/F3Net/tree/master/eval)(Matlab Version).
* [Saliency-Evaluation-Toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox)(Matlab Version).
* [SOCToolbox](https://github.com/mczhuge/SOCToolbox).
* [ssim/iou loss](https://github.com/xuebinqin/BASNet).

## Cite
If our work has helped you, you can cite us, or the above linked article.