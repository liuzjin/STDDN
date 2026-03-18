# STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation


we propose a Spatio-Temporal Decoupled Differential Equation Network (STDDN) for crowd simulation. This framework introduces the continuity equation from fluid mechanics as a physical constraint to model the dynamic evolution of crowd density from a macroscopic perspective.

## Installation
### Environment
- os: Linux
- Python = 3.8
- Pytorch = 2.4.1

To build the environment, run:
```bash
conda env create -f environment.yml
```

## Dataset
We conduct crowd simulation evaluation experiments of the model on four open-source datasets: the GC, UCY, ETH, HOOTEL.
The data is located in ./data_origin.

## Usage


To test the model, use the following command:
```bash
python main.py --dataset gc
python main.py --dataset ucy
python main.py --dataset eth
python main.py --dataset hotel
```

To test the model, use the following command:
```bash
python main.py --dataset gc --mode train
python main.py --dataset ucy --mode train
python main.py --dataset eth --mode train
python main.py --dataset hotel --mode train
```
More parameters can be tuned in the corresponding YAML configuration files.

