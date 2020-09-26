# Causal Intervention for Weakly Supervised Semantic Segmentation 
The main code for:

Causal Intervention for Weakly Supervised Semantic Segmentation, Dong Zhang, Hanwang Zhang, Jinhui Tang, Xiansheng Hua, and Qianru Sun, NeurIPS 2020. [[CONTA]](xxx)

## Requirements
* Python 3.6, pytorch 1.2.0, torchvision 0.4.0, and more in requirements.txt
* PASCAL VOC 2012 devkit and COCO
* 8 NVIDIA GPUs with more than 1024MB of memory

## Usage

### Install python dependencies
```
pip install -r requirements.txt
```

### Download PASCAL VOC 2012 devkit and COCO
* PASCAL VOC 2012 in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
* COCO in https://cocodataset.org/

### To generate pseudo_mask:

```
cd pseudo_mask
```

```
python run_sample.py
```
* You can either mannually edit the file, or specify commandline arguments.
* Replace the ground_truth annotation in PASCAL VOC 2012 with the generated pseudo_mask.

### To train DeepLab-v2:

```sh
cd segmentation
```

```sh
python main.py train --config-path configs/voc12.yaml
```

### To evaluate the performance on validation set:

```sh
python main.py test --config-path configs/voc12.yaml \
    --model-path data/models/voc12/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth
```

### To re-evaluate with a CRF post-processing:<br>

```sh
python main.py crf --config-path configs/voc12.yaml
```

### Inference Demo

To process a single image:

```bash
python demo.py single \
    --config-path configs/voc12.yaml \
    --model-path model.pth \
    --image-path image.jpg
```

To run on a webcam:

```bash
python demo.py live \
    --config-path configs/voc12.yaml \
    --model-path model.pth
```

## References

1. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image
Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*,
2018.<br>
[Project](http://liangchiehchen.com/projects/DeepLab.html) /
[Code](https://bitbucket.org/aquariusjay/deeplab-public-ver2) / [arXiv
paper](https://arxiv.org/abs/1606.00915)

2. H. Caesar, J. Uijlings, V. Ferrari. COCO-Stuff: Thing and Stuff Classes in Context. In *CVPR*, 2018.<br>
[Project](https://github.com/nightrome/cocostuff) / [arXiv paper](https://arxiv.org/abs/1612.03716)

3. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object
Classes (VOC) Challenge. *IJCV*, 2010.<br>
[Project](http://host.robots.ox.ac.uk/pascal/VOC) /
[Paper](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)

4. Ahn, Jiwoon and Cho, Sunghyun and Kwak, Suha. Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations. *CVPR*, 2019.<br>
[Project](https://github.com/jiwoon-ahn/irn) /
[Paper](https://arxiv.org/abs/1904.05044)

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

## Questions
Please contact 'dongzhang@njust.edu.cn'
