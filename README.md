The implementation of paper [*Infrared Action Detection in the Dark via Cross-Attention Mechanism*](https://ieeexplore.ieee.org/abstract/document/9316950).

We investigatethe temporal action detection problem in the dark by using infrared videos. 
Our model takes the whole video as input, a Flow Estimation Network (FEN) is employed to generate the optical flow for infrared data, and it is optimized with the whole network to obtain action-related motion representations. 
After feature extraction, the infrared stream and flow stream are fed into a Selective Cross-stream Attention (SCA) module to narrow the performance gap between infrared and visible videos. 
The SCA emphasizes informative snippets and focuses on the more discriminative stream automatically. 
Then we adopt a snippet-level classifier to obtain action scores for all snippets and link continuous snippets into final detections.
All these modules are trained in an end-to-end manner.
Experimental results show that our proposed method surpasses state-of-the-art temporal action detection methods designed for visible videos, and it also achieves the best performance compared with other infrared action recognition methods on both InfAR and Infrared-Visible datasets.

## Approach
![overview](https://github.com/LannCX/InfDetNet/blob/main/asset/overview.jpg)

## Dataset
We collect a new infrared dataset for temporal action detection in the dark, named InfDet. 
To get thermal infrared videos, we mainly consider the night scenes at distance, which is intractable both for RGB and ordinary
NIR(Near Infrared) cameras.
![dataset](https://github.com/LannCX/InfDetNet/blob/main/asset/dataset.jpg)
> Please contact *gaocq@cqupt.edu.cn* for the authority of the InfDet dataset.

## Requirements
Our code has been tested on Ubuntu16.04 using python3.6, Pytorch version 1.4.0 with four NVIDIA Tesla V100 cards.

## Run

```
# Plaese modify the dataset directory in your machine correctly.
# Train
python train.py -train True

# Test
## step 1: run detections.
python train.py -train False

## step 2: run post-processing to generate a file named "sel_i3d.txt".
python post_process.py

## step 3: evaluate results using the offical codebase provided by the authors of THUMOS'14.
cd THUMOS14_evalkit_20150930
matlab TH14evalDet('sel_i3d.txt','annotation','test',0.5)
```

## Citation
If you find our code useful in your work, please consider using the following citation:
```
@ARTICLE{chen2021infdet,  
title={Infrared Action Detection in the Dark via Cross-Stream Attention Mechanism},
author={Xu Chen and Chenqiang Gao and Chaoyu Li and Yi Yang and Deyu Meng},
year={2021},
journal={IEEE Transactions on Multimedia},
doi={10.1109/TMM.2021.3050069}
}
```

## Aknowdlegement
We thank the following nice repos used in our code:
- [https://github.com/piergiaj/tgm-icml19](https://github.com/piergiaj/tgm-icml19)
- [https://github.com/wzmsltw/pytorch-OpCounter](https://github.com/wzmsltw/pytorch-OpCounter)
