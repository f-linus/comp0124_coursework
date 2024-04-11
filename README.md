# Ant Colony Simulator

## Overview

This repository contains the code for a custom-built Python simulator designed to model and analyze the foraging behavior of ant colonies using pheromone trail deposition. The simulator offers a controlled environment to study the scaling effects of colony size on food retrieval rates. It provides insights into ant behavior which are pertinent to both biological studies and applications in robotic systems.

## Features

- Customizable Simulation Environment: Define the dynamics of the environment including pheromone decay, food spawning, and ant interactions.
- Pheromone Trail Visualization: Uses OpenCV for real-time visualization of pheromone trails and ant movement.
- Data Aggregation: Utilizes Pandas for compiling and analyzing simulation results.
- Hyperparameter Tuning: Includes scripts for optimizing simulation settings to achieve realistic ant behavior.

## Getting Started

### Prerequisites

Ensure you have Python 3.11 installed on your system. The simulator also requires additional Python packages which can be installed via Poetry or pip.

### Installation

Clone the repository:

``` bash
git clone https://github.com/f-linus/comp0124_coursework
cd comp0124_coursework
```

Install dependencies:

``` bash
poetry install
```

or

``` bash
pip install -r requirements.txt
```

### Running Simulations

Start a simulation:

``` bash
python comp0124_coursework/experiment.py
```

## Simulation Modules
- base_environment.py: Manages the simulation environment including pheromone layers, obstacles, and food sources.
- ant.py: Defines the behavior of individual ants within the colony.
- experiment.py: Orchestrates the running of simulations and the analysis of results.

## References

Aswale, Ashay, et al. "Hacking the colony: on the disruptive effect of misleading pheromone and how to defend against it." arXiv preprint arXiv:2202.01808 (2022).
