# REPORT
Released code of our ICASSP'23 oral paper: [Inductive Relation Prediction from Relational Paths and Context with Hierarchical Transformers](https://arxiv.org/abs/2304.00215)

## Requirements
In short, we use ```networkx==2.7``` and ```pandas==1.4.1``` for data preprocessing, and use ```pytorch=1.10.0``` to run the experiments. You can set the environment accordingly, or use the same environment as ours by running 
```bash
pip install -r requirements.txt
```

## Data Preprocess
run ```preprocess_data.sh``` to preprocess raw data from ```.\data_raw```.
```bash
bash preprocess_data.sh
```

## Run  Experiments
Config hyperparameters in `run.py`, then run it: 
```bash
python run.py --task WN18RR_v1 --cuda_id 0
```
We use [Visualizer](https://github.com/luo3300612/Visualizer) to visualize the attention map and obtain the explanations of REPORT results in our paper.


## Contact and Citations
For any questions about the paper or the code, please contact the first author or leave issues. If you find our code or paper useful, please consider citiing our paper as:
```bash
@INPROCEEDINGS{li2023inductive,
  author={Li, Jiaang and Wang, Quan and Mao, Zhendong},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Inductive Relation Prediction from Relational Paths and Context with Hierarchical Transformers}, 
  year={2023},
  doi={10.1109/ICASSP49357.2023.10096502}}
```
