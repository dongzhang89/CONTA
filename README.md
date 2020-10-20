# Causal Intervention for Weakly Supervised Semantic Segmentation 
The main code for:

Causal Intervention for Weakly Supervised Semantic Segmentation.
Dong Zhang, Hanwang Zhang, Jinhui Tang, Xiansheng Hua, and Qianru Sun.
NeurIPS, 2020. [[CONTA]](https://arxiv.org/abs/2009.12547)

## Requirements

* PyTorch 1.2.0, torchvision 0.4.0, and more in requirements.txt
* PASCAL VOC 2012 devkit and COCO 2014
* 8 NVIDIA GPUs, and each has more than 1024MB of memory

## Usage

### Install python dependencies

```
pip install -r requirements.txt
```

### Download PASCAL VOC 2012 and COCO

* PASCAL VOC 2012 in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
* COCO 2014 in https://cocodataset.org/

### To generate pseudo_mask:

For pseudo-mask generaction, we follow the method in [IRNet](https://arxiv.org/abs/1904.05044) without the instance-wise step.

```
cd pseudo_mask & python run_sample.py
```
* You can either mannually edit the file, or specify commandline arguments.
* Remember to replace the ground_truth annotation in PASCAL VOC 2012 with the generated pseudo_mask.

### To train the supervised semantic segmentation model:

```
cd segmentation & python main.py train --config-path configs/voc12.yaml
```

### To evaluate the performance on validation set:

```
python main.py test --config-path configs/voc12.yaml \
    --model-path data/models/voc12/deeplabv2_resnet101_msc/train_aug/final_model.pth
```

### To re-evaluate with a CRF post-processing:<br>

```
python main.py crf --config-path configs/voc12.yaml
```

### Common setting:

* Model: DeepLab v2 with ResNet-101 backbone. Dilated rates of ASPP are (6, 12, 18, 24). Output stride is 8 times.
* GPU: All the GPUs visible to the process are used. Please specify the scope with CUDA_VISIBLE_DEVICES=0,1,2,3.
* Multi-scale loss: Loss is defined as a sum of responses from multi-scale inputs (1x, 0.75x, 0.5x) and element-wise max across the scales. The unlabeled class is ignored in the loss computation.
* Learning rate: Stochastic gradient descent (SGD) is used with momentum of 0.9 and initial learning rate of 2.5e-4. Polynomial learning rate decay is employed; the learning rate is multiplied by ```(1-iter/iter_max)**power``` at every 10 iterations.
* Monitoring: Moving average loss (average_loss in Caffe) can be monitored in TensorBoard.
* Preprocessing: Input images are randomly re-scaled by factors ranging from 0.5 to 1.5, padded if needed, and randomly cropped to 321x321.
* You can find more useful tools in /tools/xxx.


### Training batch normalization

This codebase only supports DeepLab v2 training which freezes batch normalization layers, although
v3/v3+ protocols require training them. If training their parameters on multiple GPUs as well in your projects, please
install [the extra library](https://hangzhang.org/PyTorch-Encoding/) below.

```bash
pip install torch-encoding
```

Batch normalization layers in a model are automatically switched in ```libs/models/resnet.py```.

```python
try:
    from encoding.nn import SyncBatchNorm
    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d
```

### Inference Demo

To process a single image:

```
python tools/demo.py single \
    --config-path configs/voc12.yaml \
    --model-path model.pth \
    --image-path image.jpg
```

To run on a webcam:

```
python tools/demo.py live \
    --config-path configs/voc12.yaml \
    --model-path model.pth
```

## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@InProceedings{dong_2020_conta,
author = {Dong, Zhang and Hanwang, Zhang and Jinhui, Tang and Xiansheng, Hua and Qianru, Sun},
title = {Causal Intervention for Weakly Supervised Semantic Segmentation},
booktitle = {NeurIPS},
year = 2020
}
```

## References

1. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image
Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*,
2018.<br>
[Project](http://liangchiehchen.com/projects/DeepLab.html) /
[Code](https://bitbucket.org/aquariusjay/deeplab-public-ver2) / 
[Paper](https://arxiv.org/abs/1606.00915)

2. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object
Classes (VOC) Challenge. *IJCV*, 2010.<br>
[Project](http://host.robots.ox.ac.uk/pascal/VOC) /
[Paper](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)

3. Ahn, Jiwoon and Cho, Sunghyun and Kwak, Suha. Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations. *CVPR*, 2019.<br>
[Project](https://github.com/jiwoon-ahn/irn) /
[Paper](https://arxiv.org/abs/1904.05044)

## TO DO LIST

* Training code for MS-COCO
* Code refactoring
* Release the checkpoint

## Questions

Please contact 'dongzhang@njust.edu.cn'