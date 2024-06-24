# Towards autonomous resource management: <br> Deep learning prediction of CPU-GPU load balancing.

This project proposes a deep learning based load balancer for a CPU+GPU heterogeneous systems. It takes as input an OpenCL kernel in [ProGraML](https://github.com/ChrisCummins/ProGraML) format and some runtime information (the work group size and the input size) and it is able to predict the amount of work that the CPU should receive.

The load balancer is based on the heterogeneous device mapping model presented in ProGraML. ProGraML exhibited a very poor performance in our setup, which made it impossible to do any kind of experiment. Hence, in order to solve this performance issue, we decided to migrate the full ProGraML project to the pytorch-geometric library. After that, we adapted the heterogeneous device mapping model for the load balancing task. For more details, check the [master's thesis](https://github.com/igabirondo16/pyg_programl/blob/main/master_thesis.pdf).

This project is the final master's thesis of the [Master Program in Robotics, Graphics and Computer Vision](https://eina.unizar.es/mrgcv), and it has been carried out in the [The Computer Architecture Group of Zaragoza (gaZ)](https://i3a.unizar.es/en/grupos-de-investigacion/gaz). Many thanks to [Alejandro Valero Breso](https://i3a.unizar.es/en/node/134), [Rubén Gran Tejero](https://i3a.unizar.es/es/investigadores/ruben-gran-tejero) and [Darío Suárez Gracia](https://i3a.unizar.es/es/investigadores/dario-suarez-gracia) for supervising the work.

Apart from the load balancer itself, this repository holds the scripts for running the dataflow and heterogeneous device mapping models described in [ProGraML: Program Graphs for Machine Learning](https://github.com/ChrisCummins/ProGraML).


## Installation and Requirements
### Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.8 or higher
- Pip package manager
- Git

### Installation Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/igabirondo16/pyg_programl.git
    cd pyg_programl
    ```

2. (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install .
    ```

## Usage

This project is a deep-learning based load balancer for a CPU+GPU system. By the moment, it only computes the amount of work that the CPU should receive, but it does not the actual computation.

In addition, the repository contains all the different load balancing that we tested in this master's thesis.

Finally, this project can reproduce the dataflow and heterogeneous device mapping experiments presented in [ProGraML: Program Graphs for Machine Learning](https://github.com/ChrisCummins/ProGraML).

**Note:** All the scripts have their input parameters explained. To see all the possible options please run the script with the `--help` flag. For instance:

```bash
python3 main_programl_cross_validation.py --help
```

### Running the load balancer

For running the load balancer, execute:
```bash
python3 main_opencl_lb_both_devices.py --input_kernel_path ./kernels/test.cl --input_size 256 --wgsize 32 # -- discretize_problem
```

### Running the experiments of the load balancing ablation study

The different load balancing model configurations of the ablation study can be tested using:

```bash
python3 main_programl_cross_validation.py --model_name default --pooling skips --dataset_name nvidia --discretize_problem # This is for classification
```

```bash
python3 main_programl_cross_validation.py --model_name default --pooling skips --dataset_name nvidia # This is for regression
```

### Running the dataflow experiments

Training the dataflow models:

```bash
python3 main_programl_dataflow.py --dataset_name liveness --output_filename ./liveness_train.json
```

Testing the dataflow models:
```bash
python3 main_programl_dataflow.py --dataset_name liveness --test --from_checkpoint --output_filename ./liveness_test.json
```

**Note:** These experiments need > 60GB of memory, take into account your system's specifications before launching them.

### Running the heterogeneous device mapping experiment

Running the heterogeneous device mapping model:

```
python3 main_programl_cross_validation.py --model_name default --pooling default --devmap --hidden_size 32 --dataset_name nvidia
```


## Contributing

Pull requests are more than welcome! Before making any contribution, please read [CONTRIBUTING.md](https://github.com/igabirondo16/pyg_programl/blob/main/CONTRIBUTING.md).

## License

[Apace License 2.0](https://github.com/igabirondo16/pyg_programl/blob/main/LICENSE)