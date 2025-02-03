# Lachesis

<div align='center'>
<img src='./assets/Lachesis_logo_Dalle.png'
width="300px"/>
</div>

> This artifact accompanies the paper **_Lachesis: Predicting LLM Inference Accuracy using Structural Properties of Reasoning Paths_** accepted to DeepTest'25, a workshop at ICSE'25.

# Guide to Reproduction
## 0. Raw Data Files
* `AutoFL/data/*`: Contains failing tests and covered code snippet data from BugsInPy and Defects4J. All data is sourced from the AutoFL repository (https://github.com/coinse/autofl).
* `AutoFL/results/*` and `AutoFL/combined_fl_results/*`: Results from AutoFL. Iteration 1 through 5 are obtained from the AutoFL repository, while iterations 6 through 10 are generated through direct execution.

## 1. Representation of Reasoning Paths
* `AutoFL/name_utils.py`: Includes functions for processing arguments. This file was sourced from the AutoFL repository to ensure consistent preprocessing with AutoFL.
* `data/*`: LIM and LIG data represented using various embedding methods.
* `represent_data.py`: Code for generating LIM and LIG data.
To obtain the dataset representing reasoning paths, please excute the following commend:
```
python represent_data.py
```

## 2. Reproduce Results in the Paper
* `final_gcn.ipynb`, `final_lstm.ipynb`, `get_baselines.ipynb`: Experimental code for training models and calculating final results.