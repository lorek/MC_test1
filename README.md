# Lectures on Monte Carlo Theory


|  Book title:       |Lectures on Monte Carlo Theory  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;              |
|--------------------|------------------------------------------------|
| **Authors**        | Paweł Lorek and Tomasz Rolski                 |
| **Affiliation**    | University of Wrocław                         |
| **Publisher**      | Springer, 2025                                |
| **Series**         | *Probability Theory and Stochastic Modelling* |



This repository contains Python scripts and code examples for the Monte Carlo simulations presented in the above book.

 
## About the Code

Most examples should run out-of-the-box, provided you are using Python 3.12. The numerical values and figures in the book can be reproduced directly from the scripts. All required packages are listed in `requirements.txt`.


To ensure full compatibility, it is recommended to use the exact package versions provided, preferably by setting up the environment via Conda (see instructions below). If you encounter issues, you can also use the provided Docker image for a fully reproducible setup.


## Prerequisites
- Python 3.12.2
- Conda (recommended), pip, or Docker




## Installation
### 1. Using Conda (Recommended)

If you don’t have Conda installed, we recommend installing **Miniconda**, a lightweight version of Anaconda:

- Download **Miniconda** for your system: https://docs.conda.io/en/latest/miniconda.html
- Follow the installation instructions for your operating system.
  On Linux/macOS, restart your terminal after installation.
  On Windows, open the **Anaconda Prompt** after installation.

We provide two versions of the Conda environment:

- `environment.yml` – minimal version with essential packages
- `environment_full.yml` – full version capturing the entire working environment used during development

To create and activate the environment:

```bash
conda env create -f environment.yml
conda activate MonteCarlo_book
```

Or to use the full version for maximum compatibility:

```bash
conda env create -f environment_full.yml
conda activate MonteCarlo_book
```

> ⚠️ In rare cases, when you experience missing packages, layout issues, or unexpected errors,
> we recommend using `environment_full.yml` to ensure complete reproducibility across systems.


### 2. Using pip
```bash
pip install -r requirements.txt
```

## Running Examples
To execute a script, use:
```bash
python ch1_monte_carlo.py
```

## Repository Structure
```
book_code/
├── chapter_1/
│   ├── ch1_2d_game_winning_prob.py
│   ├── ch1_estimate_pi_quasirandom.py
│   ├── ch1_simple_random_walk.py
│
├── chapter_2/
│   ├── ch2_random_walk.py
│   ├── ch2_prng_sets_A_B_C.py
...
├── chapter_8/
│   ├── ch2_aloha.py
│   ├── ...
│
└── requirements.txt
└── environment.yml
└── README.md
```

## Running Scripts and Reproducing Results

To run a script, navigate to its folder and use:

```bash
python ch1_monte_carlo.py
```

This assumes you are inside the corresponding directory, such as `chapter_1/`.

When run without arguments, each script uses the same parameters and random seed as in the book, so it reproduces the reported results and figures.

The output is typically displayed on screen, and figures are also saved as `.pdf` files in a subdirectory called `results/`. This folder is created automatically if it does not exist.

Most scripts support command-line arguments that allow you to explore different settings. To see the available options, run:

```bash
python ch1_monte_carlo.py --help
```

**Example:** the following command runs the script `ch5_strat_sampling_pi.py` (which estimates $\pi$ using stratified sampling) with 500 replications, 4 strata, and a custom seed:

```bash
python ch5_strat_sampling_pi.py --R 500 --m 4 --seed 54321
```

You can also pass comma-separated lists of values, e.g.,
```bash
python ch5_strat_sampling_pi.py
```

is equivalent to the following explicit call (these default parameters can also be seen using the `--help` option):

```bash
python ch5_strat_sampling_pi.py --R 200,10000 --Rpilot 100 --m 5,10,20 --results_path results --seed 31415
```
which estimates $\pi$ several times: using 200 and 10000 replications, and 5, 10, or 20 strata.


##   Full reproducibility with Docker (Optional)

In rare cases, if installing with Conda does not fully reproduce the expected results (e.g., due to subtle version differences or system dependencies), you can run all code examples inside a fully configured Docker container.

See [README_docker.md](README_docker.md) for full instructions.
 
## License
This code is provided under the MIT License.
