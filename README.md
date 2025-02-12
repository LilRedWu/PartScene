Mask3D is a deep learning-based framework for 3D part segmentation. This repository contains code for training, evaluating, and utilizing 3D segmentation models.

Installation

Prerequisites

Ensure you have the following installed:

Python 3.8+

CUDA (for GPU acceleration)

PyTorch (compatible with your CUDA version)

Git LFS (for handling large files)

Setup

Clone the repository:

git clone https://github.com/LilRedWu/Mask3D.git
cd Mask3D

Install dependencies:

pip install -r requirements.txt

Set up large files using Git LFS:

git lfs install
git lfs pull

Usage

Preprocessing Data

Before training, ensure your dataset is correctly formatted. You may need to preprocess part masks using:

python preprocess_partmask.ipynb

Training the Model

To train the 3D part segmentation model, run:

python train.py --config configs/config.yaml

Modify the config file as needed.

Evaluation

Run the evaluation script on a trained model:

python evaluate.py --model checkpoints/demo/model.pth

Running Inference

To perform inference on new data:

python inference.py --input input_file.ply --output output_file.ply

Project Structure

Mask3D/
│── build/                   # Compiled dependencies
│── checkpoints/demo/        # Pre-trained models
│── model/                   # Model architectures
│── output/                  # Output results
│── point_sam/               # Point-based SAM integration
│── scripts/                 # Helper scripts
│── third_party/             # External dependencies
│── utils/                   # Utility functions
│── benchmark.sh             # Benchmarking script
│── evaluate.py              # Evaluation script
│── mask_classification.py   # Mask classification module
│── part_seg_2d_ppl.py       # 2D segmentation pipeline
│── part_seg_all.ipynb       # Notebook for segmentation
│── part_seg_ppl.py          # Removed script
│── preprocess_partmask.ipynb # Data preprocessing notebook

Acknowledgments

This repository builds upon various open-source projects for 3D deep learning. Contributions and modifications have been made to adapt it for specific tasks.

Contact

For any issues or questions, please contact the repository owner or raise an issue in the GitHub repository.
