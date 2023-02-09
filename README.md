# MDRNet+ ---PyTorch
This is the official PyTorch implementation of MDRNet+ (IEEE TNNLS 2023): [Mitigating Modality Discrepancies for RGB-T Semantic Segmentation](https://ieeexplore.ieee.org/document/10008228)
## Introduction
MDRNet+ is an improved and expanded version of our CVPR 2021 paper [ABMDRNet: Adaptive-weighted Bi-directional Modality Difference Reduction Network for RGB-T Semantic Segmentation](https://ieeexplore.ieee.org/document/9578077). An earlier version of this article was presented in part at CVPR 2021, DOI: 10.1109/CVPR46437.2021.00266
## Dataset
MFNet dataset: Original dataset [download](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/), RTFNet version [download](https://github.com/yuxiangsun/RTFNet), GMNet version [download](https://github.com/Jinfu0913/GMNet), our model use the MFNet dataset provided by GMNet
## Notice
Note that change the dataset path as well as the save weight path in ./util/util, ./util/irseg, ./train.py
## Pretrained weights
The final weights used in this paper: [download](https://pan.baidu.com/s/1wolrOPNvQrOrIQtrwGm_jw) &ensp;&ensp;password：4kod
## Results
Predict maps: [download](https://pan.baidu.com/s/1KL65FQBue8Q5MIdayE-7Sg) &ensp;&ensp;password：pxhg
## Citation
If you find ABMDRNet and MDRNet+ useful in your research, please consider citing:  
@inproceedings{zhang2021abmdrnet,  
              &ensp; &ensp;title={ABMDRNet: Adaptive-weighted bi-directional modality difference reduction network for RGB-T semantic segmentation},  
              &ensp; &ensp;author={Zhang, Qiang and Zhao, Shenlu and Luo, Yongjiang and Zhang, Dingwen and Huang, Nianchang and Han, Jungong},  
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},  
  pages={2633--2642},  
  year={2021}  
}  
@article{zhao2023mitigating,  
  title={Mitigating Modality Discrepancies for RGB-T Semantic Segmentation},  
  author={Zhao, Shenlu and Liu, Yichen and Jiao, Qiang and Zhang, Qiang and Han, Jungong},  
  journal={IEEE Transactions on Neural Networks and Learning Systems},  
  year={2023},  
  publisher={IEEE}  
}
## Acknowledgement
Thanks the code of [RTFNet](https://github.com/yuxiangsun/RTFNet) provided by Yuxiang Sun and [GMNet](https://github.com/Jinfu0913/GMNet)!
## Contact
Please drop me an email for further problems or discussion: zhaoshenlu@stu.xidian.edu.cn
