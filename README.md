# Reinforcement Learning Project: MuJoCo Hopper and Panda-Gym PushTask

This repository contains a reinforcement learning project developed for the course **Fundamentals of Artificial Intelligence, Machine Learning and Deep Learning** at Politecnico di Torino.

The project focuses on training, evaluating, and analyzing reinforcement learning agents in simulated control environments. The main environments considered are **MuJoCo Hopper** and **Panda-Gym PushTask**.

The work includes environment setup, random policy testing, agent training, evaluation, reward analysis, and technical reporting.

## Project Overview

The goal of the project is to study reinforcement learning methods for control tasks in simulation-based environments.

The repository is divided into two main parts:

- **Part 1 — MuJoCo Hopper**  
  Experiments on the Hopper continuous-control environment using Gymnasium and MuJoCo.

- **Part 2 — Panda-Gym PushTask**  
  Experiments on a robotic pushing task using Panda-Gym.

## Main Topics

- Reinforcement Learning
- Continuous Control
- Policy Optimization
- Simulation-Based Learning
- Agent Evaluation
- Reward Analysis
- Domain Randomization
- Robotics Simulation

## Tools and Libraries

- Python
- Gymnasium
- MuJoCo
- Stable-Baselines3
- Panda-Gym
- NumPy
- Matplotlib
- Jupyter Notebook

## Repository Structure
.
├── part1/                         # MuJoCo Hopper experiments
├── part2/                         # Panda-Gym PushTask experiments
├── Project_RL.pdf              # Technical project report
├── requirements.txt               # Python dependencies
└── README.md

Report

The full technical report is available here:

Reinforcement Learning Project Report⁠￼

The report describes the methodology, implementation choices, experiments, results, and limitations of the project.

How to Run

Install the required dependencies:
pip install -r requirements.txt

Run the initial random-policy test for Part 1:
cd part1
python test_random_policy.py

For the Panda-Gym task:
cd part2/panda-gym
pip install -e .

Then run the relevant training or evaluation scripts from the project folders.

Skills Demonstrated

* Reinforcement learning experimentation
* Python-based machine learning workflows
* Simulation environment setup
* Agent training and evaluation
* Reward curve analysis
* Technical reporting
* Scientific computing
