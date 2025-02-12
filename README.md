
# PartScene - 3D Part Segmentation

This paper aims to achieve the segmentation of any 3D part based on natural language descriptions, extending beyond traditional object-level 3D scene understanding and addressing both data and methodological challenges. Existing datasets and methods are predominantly limited to object-level comprehension.

To overcome the limitations of data availability, we introduce the first large-scale 3D dataset with dense part annotations, created through an innovative and cost-effective method for constructing 3D scenes with fine-grained part-level annotations, paving the way for advanced 3D part understanding.

On the methodological side, we propose a two-stage “search & localize” strategy to effectively tackle the challenges of part-level segmentation. Extensive experiments demonstrate the superiority of our approach in open-vocabulary 3D understanding tasks at both the part and object levels, with strong generalization capabilities across various 3D datasets.
For more details, please check our [paper](https://lilredwu.github.io/).

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


## Steup
### Preprocess the 3d mask proposal 
python mask_classification.py 


### Preprocess the 3d-2d pair 
python part_seg_2d_ppl.py


### Train Model

python train.py --config configs/train_config.yaml


### Evaluate Model
python evaluate.py --model_path path_to_checkpoint


###  Running Benchmark

bash benchmark.sh


#Contributions
If you'd like to contribute, feel free to submit a pull request or open an issue for discussion.
