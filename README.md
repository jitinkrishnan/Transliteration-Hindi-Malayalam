## Cross-Lingual Text Classification of Transliterated Hindi and Malayalam

Source Code for the paper "Cross-Lingual Text Classification
of Transliterated Hindi and Malayalam"

### Paper/Cite
https://arxiv.org/abs/2108.13620 (To appear at [IEEE Big Data 2022](https://bigdataieee.org/BigData2022/))
```
@article{krishnanDiversity,
  title={Cross-Lingual Text Classification of Transliterated Hindi and Malayalam},
  author={Krishnan, Jitin and Anastasopoulos, Antonios and Purohit, Hemant and Rangwala, Huzefa},
  journal={In Proceedings of IEEE International Conference on Big Data},
  year={2022}
}
```

### Requirements
- pip install transformers
- Enable Cuda

### How to Run: Baseline Models

```python3 <filename>.py <target> <base_line_type> <mlm_model_name>```

filename: ```{mlm_floods, mlm_movie}```
target: ```{ml, hi}```
base_line_type: ```{en, trt, tlt, combo}```
mlm_model_name: ```{xlm-roberta-base, bert-base-multilingual-cased}```

### How to Run: Joint-TS (Teacher-Student) Model

```python3 <filename>.py <target> <gamma> <mlm_model_name>```

filename: ```{mlm_joint_ts_floods, mlm_joint_ts_movie}```
target: ```{ml, hi}```
gamma: ```{0.01-1.0}```
mlm_model_name: ```{xlm-roberta-base, bert-base-multilingual-cased}```


### Constructing Tweet Data files

Get tweet text using the TWEETIDS provided as CSV files in the
data folder, and construct the following text files:

- Kerala Floods Tweets: ```kf_pos```, ```kf_neg```
- North India Floods Tweets: ```ni_pos```, ```ni_neg```

### Contact information
For help or issues, please submit a GitHub issue or contact Jitin Krishnan (`jkrishn2@gmu.edu`).
