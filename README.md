# Transformer KV-Cache Assignment

# Overview
This assignment explores how memory systems and caching affects inference performance in Transformer language models.
The three main focuses are Key-Value cache, batching, and precision using a small pretrained model (DistilGPT-2). 

The assignment is divided into three main parts:
-**Part A:** Computer Architecture & Cache Fundamentals
-**Part B:** KV-Cache Theory and Analysis
-**Part C:** Coding and Experimental Evaluation

## Setup Instructions
## Requirements
Before running the Experiments, make sure Python and the required packages are installed.

```
```bash
pip install torch transformers psutil

## Running Experiments

Each Experiment can be executed individually from the terminal

```bash
# Pretrained model
python3 run_model.py
```

# Experiment 1 - With and Without KV Cache
python3 experiment1.py

# Experiment 2 - Batching Effect
python3 experiment2.py

# Experiment 3 - Precision
python3 experiment3.py

All results are saved in the results/ folder as .csv files.



Summary of Results:

Experiment 1:
use_cache	Avg Time (s)	Tokens/sec	Memory (MB)
True	       2.97	         17.5	       ≈680
False	       2.43        	20.6        	≈680

Observations:
Without cache, inference was slightly faster since the sequence was short. However,
KV caching becomes crucial in longer sequences, where it prevents uneccessary computations and improves efficiency.


Experiment 2:
Batch size	Elapsed Time (s)	Tokens/sec	  Memory before	     Memory after 
    1	           3.317	        15.07	       370180096	        701136896
    2	           2.615	        19.12	       701267968        	709525504
    4	           3.767	        13.27	       709525504         	738361344

Observations: Batching improves throughput up to an extent, but also increases memory usage.
Batch size 2 achieved the best balance between efficiency and performance.

Experiment 3:
Precision	  Elapsed Time (s)	Tokens/sec	Memory before  	Memory After
  FP32	         6.762          7.39	     372379648	     704647168
  FP16	         10.368	         4.82	     528580608	     530153472

Observation: On CPU, FP32 performed faster and FP16 provides better memory efficiency and lower
power usage.


AUTHOR:
Saray Alvarado
GitHub: @sarraayy











