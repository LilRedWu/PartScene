
# PartScene - 3D Part Segmentation

This repository provides a deep learning-based framework for 3D part segmentation. PartScene utilizes advanced neural architectures to process and segment point cloud data effectively.

For more details, please check our [paper]().

## System Requirements

- cuda=11.7
- python 3.8
- gcc=7.5.0
- torch=1.13

## Package Requirements

```bash
conda create -n partscene_env python=3.8
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Compile the CUDA dependencies using the following command:

```bash
cd utils/emd_ && python setup.py install 
```

## Preparation

Datasets are available [here](https://drive.google.com/drive/folders/1dAH9R3XDV0z69Bz6lBaftmJJyuckbPmR). Run the command below to download all datasets (ShapeNetRender, ModelNet40, ScanObjectNN, ShapeNetPart) to reproduce the results.

```bash
cd Data
source download_data.sh
```

## Unsupervised Training

```bash
python train.py --model pointnet --dataset Data/ --nepoch 100 --dataset_type shapenet --lr 0.001 --decay 0.8 --step_size 2000 --batchSize 64 --task_type segmentation --structure 3layer --feature_size 128 --regularization chamfer
```

## Supervised Training

Firstly, load the model parameters trained in contrastive learning. Then, generate the deform dataset using the Deform Net:

```bash
python scripts/get_dataset.py --deform_net1_path path/to/deform_net_1.pth.tar --deform_net2_path path/to/deform_net_2.pth.tar --classifier_path path/to/best_model.pth.tar --dataset_path /path/to/dataset
```

Second, use the augmented data for supervised training:

```bash
cd third_party/Pointnet_Pointnet2_pytorch
```

```bash
python train_classification.py --log_dir pointnet_cls --dataset /path/to/dataset --dataset_type modelnet40_npy --epoch 50
```

## Email for QA

Any questions related to the repo, please send an email to: `1353099226why@gamil.com`.

## Acknowledgment

This repository is developed based on [https://github.com/MohamedAfham/CrossPoint.git](https://github.com/MohamedAfham/CrossPoint.git). Thanks for their contribution.

## Citation

If you find our work useful, please kindly cite our work.
