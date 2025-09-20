# Intelligent Fish Detection System with Similarity-Aware Transformer 

### Shengchen Li, Haobo Zuo, Changhong Fu, Zhiyong Wang, Zhiqiang Xu

## Abstract
Fish detection in water-land transfer has significantly contributed to the fishery.
However, manual fish detection in crowd-collaboration performs inefficiently and expensively, involving insufficient accuracy.
To further enhance the water-land transfer efficiency, improve detection accuracy, and reduce labor costs, this work designs a new type of lightweight and plug-and-play edge intelligent vision system to automatically conduct fast fish detection with high-speed camera. 
Moreover, a novel similarity-aware vision Transformer for fast fish detection (FishViT) is proposed to onboard identify every single fish in a dense and similar group.
Specifically, a novel similarity-aware multi-level encoder is developed to enhance multi-scale features in parallel, thereby yielding discriminative representations for varying-size fish. Besides, a new soft-threshold attention is presented to effectively eliminate background noise from images and precisely recognize the edge information of different similar fish.
80 challenging video sequences with high framerate and high-resolution are collected to establish a benchmark from real fish water-land transfer scenarios. Exhaustive evaluation conducted with this challenging benchmark has proved the robustness and effectiveness of FishViT with 82.3 FPS. Real work scenario tests validate the practicality of the proposed method.
![Workflow of our tracker](https://github.com/vision4robotics/FishViT/blob/main/images/1.jpg)

This figure shows the workflow of our FishViT.

## About Code
### 1. Environment setup
This code has been tested on Ubuntu 22.04, Python 3.9, Pytorch 1.13.1, CUDA 11.7.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

### 2. Test

```bash 
python detect.py                                
	--source demo/fish2736.mp4          # video path
```
The testing result will be saved in the `runs/detect/exp` directory.

### 3. Contact
If you have any questions, please contact me.

Shengchen Li

Email: [shengcli@tongji.edu.cn](shengcli@tongji.edu.cn)

For more evaluations, please refer to our paper.

## References 

```

```
