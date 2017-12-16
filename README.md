# AGNet: Attention-Guided Network for Surgical Tool Presence Detection

by Xiaowei Hu, Lequan Yu, Hao Chen, Jing Qin and Pheng-Ann Heng

This implementation is written by Xiaowei Hu at the Chinese University of Hong Kong.

***

## Citation

@incollection{hu2017agnet,   
&nbsp;&nbsp;&nbsp;&nbsp;  title={AGNet: Attention-Guided Network for Surgical Tool Presence Detection},   
&nbsp;&nbsp;&nbsp;&nbsp;  author={Hu, Xiaowei and Yu, Lequan and Chen, Hao and Qin, Jing and Heng, Pheng-Ann},   
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle={Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support},   
&nbsp;&nbsp;&nbsp;&nbsp;  pages={186--194},   
&nbsp;&nbsp;&nbsp;&nbsp;  year={2017},   
&nbsp;&nbsp;&nbsp;&nbsp;  publisher={Springer}   
}

## Installation
1. Clone the AGNet repository, and we'll call the directory that you cloned AGNet into `AGNet`.

    ```shell
    git clone https://github.com/xw-hu/AGNet.git
    ```

2. Build Caffe

   *This model is tested on Ubuntu 16.04, CUDA 8.0, cuDNN 5.0   
    
   Follow the Caffe installation instructions here: [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html)   

   ```shell
   make all -j XX
   make matcaffe
   ```
   
3. Download data in `AGNet/data/`.

## Train and Test

1. Download the pre-trained model in `AGNet/models/`.

2. Run `AGNet/matlab/AGNet_train.m` or `AGNet/matlab/AGNet_test.m`.
   
   
