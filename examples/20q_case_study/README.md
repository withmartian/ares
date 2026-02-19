## Case Study: Mech Interp for 20 Questions

This folder contains example code for using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) with [ARES](https://github.com/withmartian/ares) to conduct mechanistic interpretability research on a toy 20 Questions environment. 

### Our Findings

The accompanying notebook [ares_mi_20q_tutorial.ipynb](./ares_mi_20q_tutorial.ipynb) contains results from our experiments - open it to see read through.

### Reproducing Results

To follow along and create results yourself, run the scripts located in this directory:
- [collect_20q_data.py](./collect_20q_data.py) generates data by interacting with the 20 Questions environment.
- [phase1_probe.py](./phase1_probe.py) trains a linear probe on the collected data.
- [phase2_steer.py](./phase2_steer.py) runs the environments again but this time introducing steering vectors to affect the LLM's behavior.
- [analyse_steering.py](./analyse_steering.py) and [visualise_steering_vectors.py](./visualise_steering_vectors.py) condict analysis of the steering experiments.

After running the relevant scripts, you can also run through the notebook with the relevant data generated to reproduce the findings there.