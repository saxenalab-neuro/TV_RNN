# TV_RNN

This repo is about the work of Time Varying Recurrent Neural Networks (TV RNNs) with its analysis. Preprint available at: https://openreview.net/forum?id=0SD-uuvShZx

# Installation for Development
1. Kernel: Pytorch 1.8.1 (used for doing this experiment, not required)
2. Link to SHAP value installation: https://github.com/slundberg/shap. Skip this step if not computing SHAP
# Implementation
1. git clone this repo to your local folder
2. build environment
```
conda env create --file environment_tvrnn.yml
conda activate tv-rnn
```
3. If environment is not built successfully, do 'pip install'+all missed packages, they are common packages in this experiment. 
4. ```python demo.py``` (example running) 
5. or open demo.ipynb

# Details
1. timevarying.py: TV RNN algorithm
2. decoding.py: Decoding algorithm
3. analysis.py: Weights analysis
4. shaping.py: SHAP value computing
5. training.py: Training process
6. demo.ipynb: Demo notebook
