# Running the Code with Docker

|  Book title:       |Lectures on Monte Carlo Theory  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;              |
|--------------------|------------------------------------------------|
| **Authors**        | Paweł Lorek and Tomasz Rolski                 |
| **Affiliation**    | University of Wrocław                         |
| **Publisher**      | Springer, 2025                                |
| **Series**         | *Probability Theory and Stochastic Modelling* |



This document explains how to run all Python scripts from the `MC_test1` directory using Docker, without requiring a local installation of Python or Conda. The provided Docker setup ensures full reproducibility of the code examples and dependencies.

## 1. Prerequisites

Make sure you have Docker installed on your system.

- Install Docker: https://docs.docker.com/get-docker/
- Optional (Linux only): to avoid using `sudo` with Docker, add your user to the `docker` group:

```bash
sudo usermod -aG docker $USER
# Then restart your session (e.g., log out and log in)
```

## 2. Build the Docker Image

Clone or download this repository. Make sure the `Dockerfile` and `environment_full.yml` are in the `MC_test1` directory.

From the parent directory of `MC_test1`, run:

```bash
docker build -t montecarlo_book .
```

This command builds a Docker image named `montecarlo_book` using the full Conda environment specification.

## 3. Run the Container

To start an interactive container:

```bash
docker run -it montecarlo_book
```

This opens a terminal session inside the container, with the Conda environment `MonteCarlo_book` already activated and ready to use.

You’ll be located in the `/app` directory, which contains the `MC_test1` folder.

## 4. Running Scripts

Inside the container:

```bash
cd MC_test1/chapter_1
python ch1_2d_game_winning_prob.py
```

Some scripts generate plots saved to the `results/` directory. By default, plots are not shown interactively in Docker, but PDFs are saved.

## Notes

- All dependencies are installed using the `environment_full.yml` specification to guarantee consistent versions.
- The container does **not** mount local directories or enable GUI display by default. For advanced usage (e.g., sharing volumes or running graphical apps), additional Docker options are needed.


➡️ [Back to main README](README.md)
