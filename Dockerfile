##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##
## Dockerfile to build a fully reproducible environment for the book:       ##
## "Lectures on Monte Carlo Theory"                                         ##
## Authors: Paweł Lorek, Tomasz Rolski                                      ##
## Publisher: Springer, 2025                                                ##
## Series: Probability Theory and  Stochastic Modelling                     ##
##
##                                                                          ##
## It uses the full Conda environment specification (environment_full.yml)  ##                                                              ## to ensure exact versions of all dependencies.                            ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##


FROM continuumio/miniconda3

# Set working directory inside the container
WORKDIR /app

# Copy the entire MC_test1 directory, including environment_full.yml
COPY . /app/MC_test1

# Create the Conda environment from the full specification
RUN conda env create -f /app/MC_test1/environment_full.yml

# Ensure the environment is activated by default in future RUN/CMD
SHELL ["conda", "run", "-n", "MonteCarlo_book", "/bin/bash", "-c"]

# Set default command when container starts
CMD ["bash"]
