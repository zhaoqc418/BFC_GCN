# BVLC

## Requirements
* Python >= 3.6
* PyTorch >= 0.4.0
* [faiss](https://github.com/facebookresearch/faiss)
* [mmcv](https://github.com/open-mmlab/mmcv)


## Dataset Preparation

### Data format

The data directory is constucted as follows:
```
.
├── data
|   ├── features
|   |   └── xxx.bin
│   ├── labels
|   |   └── xxx.meta
│   ├── knns
|   |   └── ... 
```

- `features` currently supports binary file. (We plan to support `np.save` file in near future.)
- `labels` supports plain text where each line indicates a label corresponding to the feature file.
- `knns` is not necessary as it can be built with the provided functions.

Take MS-Celeb-1M (Part0 and Part1) for an example. The data directory is as follows:
```
data
  ├── features
    ├── part0_train.bin                 # acbbc780948e7bfaaee093ef9fce2ccb
    ├── part1_test.bin                  # ced42d80046d75ead82ae5c2cdfba621
  ├── labels
    ├── part0_train.meta                # class_num=8573, inst_num=576494
    ├── part1_test.meta                 # class_num=8573, inst_num=584013
  ├── knns
    ├── part0_train/faiss_k_80.npz      # 5e4f6c06daf8d29c9b940a851f28a925
    ├── part1_test/faiss_k_80.npz       # d4a7f95b09f80b0167d893f2ca0f5be5
```

To experiment with custom dataset, it is required to provided extracted features and labels.
For training, the number of features should be equal to the number of labels.
For testing, the F-score will be evaluated if labels are provided, otherwise only clustering results will be generated.

Note that `labels` is only required for training clustering model, but it is not mandatory for clustering unlabeled data.
Basically, there are two ways to cluster unlabeled data without meta file.
(1) Do not pass the `label_path` in config file. It will not generate loss and evaluation results.
(2) Make a pseudo meta label, e.g., setting all labels to -1, but just ignore the loss and the evaluation results.

###  Supported datasets
The supported datasets are listed below.

- [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)
    - Part1 (584K): [GoogleDrive](https://drive.google.com/open?id=16WD4orcF9dqjNPLzST2U3maDh2cpzxAY) or [BaiduYun](https://pan.baidu.com/s/1i4GYYNKTyp3lvOYLrvWc0g) (passwd: geq5)
    - Original Images: [OneDrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155095455_link_cuhk_edu_hk/ErY9MmAhmlZMvO9y9SagNOcBISISEdzBfJshn-poD84QPQ?e=PRRpBe). We re-align [MS1M-ArcFace](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0) with our own face alignment model.
    - Pretrained Face Recognition Model: [GoogleDrive](https://drive.google.com/open?id=1eKsh7x-RUIHhIJ1R9AlUjsJdsdbh2qim). For using the model to extract features, please check the [code](https://github.com/yl-1993/hfsoftmax/tree/ltc) and use [sample data](https://drive.google.com/open?id=1VkZWZmBnaQlTaTNQSQXe-8q8Z8pNuI-D) to have a try.
- [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html): [GoogleDrive](https://drive.google.com/open?id=15B5Ypj8_U9rhcuvkrkCZQAgV4cfes7aV) or [BaiduYun](https://pan.baidu.com/s/174XeXhCOBAMryKcz9IDc8g) (passwd: 8fai)

## Run

### Test

### Train
