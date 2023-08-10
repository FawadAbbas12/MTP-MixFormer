## Multi Target proposal MixFormer and KF-based liner occlusion handling
This is a trimmed-down version of MixFormer to continue on proposed implementation by adding the following features:

- Isolate Mixformer Convmae Online model so it is easy to integrate into other applications  ✅
- Add an auxiliary Linear motion model Kalman filter to avoid id switch for a short period   ✅
- Implement reassociation methodology and heuristic for kalman update  
- Implement center point based multi detection head
- Add an auxiliary EKF or PSO Based Object Tracking module to maintain tracklet for the original target

# MixFormer (Base REPO)
[MixFormer](https://www.votchallenge.net/howto/tutorial_python.html)

### :sparkles: Strong performance
| Tracker | VOT2020 (EAO) | LaSOT (NP)| GOT-10K (AO)| TrackingNet (NP)|
|---|---|---|---|---|
|**MixViT-L (ConvMAE)**|0.567|**82.8**|-|**90.3**|
|**MixViT-L**|**0.584**|82.2|**75.7**|90.2|
|**MixCvT**|0.555|79.9|70.7|88.9|
|ToMP101* (CVPR2022)|-|79.2|-|86.4|
|SBT-large* (CVPR2022)|0.529|-|70.4|-|
|SwinTrack* (Arxiv2021)|-|78.6|69.4|88.2|
|Sim-L/14* (Arxiv2022)|-|79.7|69.8|87.4|
|STARK (ICCV2021)|0.505|77.0|68.8|86.9|
|KeepTrack (ICCV2021)|-|77.2|-|-|
|TransT (CVPR2021)|0.495|73.8|67.1|86.7|
|TrDiMP (CVPR2021)|-|-|67.1|83.3|
|Siam R-CNN (CVPR2020)|-|72.2|64.9|85.4|
|TREG (Arxiv2021)|-|74.1|66.8|83.8|

## Install the environment
Use the Anaconda
```
conda create -n mixformer python=3.6
conda activate mixformer
bash install_pytorch17.sh
```

## Test and evaluate MixFormer on Your own video

### Only Isolated model (change vid path to your own video)
```
python mixformer_convmae_online.py 
```

### Model with Liner KF
```
from KF_tracker_wrapper import TrackingModel_2D

tracker = TrackingModel_2D()
vid_path = 'path/to/your/video'
tracker.track(vid_path)
```

## Model Zoo and raw results
The trained models and the raw tracking results are provided in the [[Models and Raw results]](https://drive.google.com/drive/folders/1wyeIs3ytYkmAtTXoVlLMkJ4aSTq5CBHq?usp=sharing) (Google Driver) or
[[Models and Raw results]](https://pan.baidu.com/s/1k819gnFMav9t1-8ZhCo74w) (Baidu Driver: hmuv).

## Contact
Fawad Abbas: fawad.abbas04@gmail.com

Yutao Cui(Orignal Autohr): cuiyutao@smail.nju.edu.cn 

## Acknowledgments
* Thanks for [MCG-NJU](https://github.com/MCG-NJU) for opensourcing their implementation 

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite author's paper:

```
@inproceedings{cui2022mixformer,
  title={Mixformer: End-to-end tracking with iterative mixed attention},
  author={Cui, Yutao and Jiang, Cheng and Wang, Limin and Wu, Gangshan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13608--13618},
  year={2022}
}
@misc{cui2023mixformer,
      title={MixFormer: End-to-End Tracking with Iterative Mixed Attention}, 
      author={Yutao Cui and Cheng Jiang and Gangshan Wu and Limin Wang},
      year={2023},
      eprint={2302.02814},
      archivePrefix={arXiv}
}
```
