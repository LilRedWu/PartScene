\documentclass{article}
\usepackage{hyperref}
\usepackage{listings}

\title{Mask3D - 3D Part Segmentation}
\author{}
\date{}

\begin{document}

\maketitle

\section{Overview}
PartScene is a deep learning-based framework for 3D part segmentation. This repository contains code for training, evaluating, and utilizing 3D segmentation models.

\section{Installation}

\subsection{Prerequisites}
Ensure you have the following installed:
\begin{itemize}
\item Python 3.8+
\item CUDA (for GPU acceleration)
\item PyTorch (compatible with your CUDA version)
\item Git LFS (for handling large files)
\end{itemize}

\subsection{Setup}
\begin{lstlisting}[language=bash]
git clone https://github.com/LilRedWu/PartScene.git
cd Mask3D
pip install -r requirements.txt
git lfs install
git lfs pull
\end{lstlisting}

\section{Usage}

\subsection{Preprocessing Data}
Before training, ensure your dataset is correctly formatted. You may need to preprocess part masks using:
\begin{lstlisting}[language=bash]
python preprocess_partmask.ipynb
\end{lstlisting}

\subsection{Training the Model}
To train the 3D part segmentation model, run:
\begin{lstlisting}[language=bash]
python train.py --config configs/config.yaml
\end{lstlisting}
Modify the config file as needed.

\subsection{Evaluation}
Run the evaluation script on a trained model:
\begin{lstlisting}[language=bash]
python evaluate.py --model checkpoints/demo/model.pth
\end{lstlisting}

\subsection{Running Inference}
To perform inference on new data:
\begin{lstlisting}[language=bash]
python inference.py --input input_file.ply --output output_file.ply
\end{lstlisting}

\section{Project Structure}
\begin{verbatim}
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
\end{verbatim}

\section{Acknowledgments}
This repository builds upon various open-source projects for 3D deep learning. Contributions and modifications have been made to adapt it for specific tasks.

\section{Contact}
For any issues or questions, please contact the repository owner or raise an issue in the GitHub repository.

\end{document}

