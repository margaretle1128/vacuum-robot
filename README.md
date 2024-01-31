# Vacuum Cleaner Search Simulation

## Description

This project is a simulation of a vacuum cleaning robot navigating a 2D grid environment to clean dirty rooms and avoid obstacles. It incorporates various search algorithms to determine the most efficient cleaning path.

## GUI Overview
<img width="557" alt="image" src="https://github.com/margaretle1128/vacuum-robot/assets/93006609/a50ce281-995f-45e1-9530-7c7b9bcc3be3">

The GUI represents the vacuum cleaner's environment with the following elements:

- **White Squares:** Clean rooms accessible to the vacuum cleaner.
- **Grey Squares:** Dirty rooms requiring cleaning.
- **Red Squares:** Obstacles that the vacuum cleaner must navigate around.
- **Orange Path:** The planned path for the vacuum cleaner, based on the chosen search algorithm.
- **Pink Squares:** Areas that have been explored during the pathfinding process.

Metrics displayed at the top of the GUI include:

- **Number-of-steps:** The cumulative count of steps taken from the start.
- **Total Cost:** The total cost incurred, factoring in movement and any active turn costs.

## Controls

- **Reset:** Initializes a new environment and restarts the simulation.
- **Next:** Advances the vacuum cleaner to the next step in its path.
- **Run:** Automates the cleaning process with a one-second step interval.
- **TurnCost:** Enables a cost for 90° turns, with each turn costing 0.5. Toggle on or off.
- **Search Type Dropdown:** Chooses the algorithm for path planning.

## Features

- Simulation of intelligent vacuum cleaning in a grid layout.
- Supports BFS, DFS, UCS, Greedy, and A* search algorithms.
- Configurable environment dimensions via command-line arguments.
- Interactive GUI with real-time visual feedback on the simulation state.

## Requirements

- Python 3.x
- Tkinter library (included with Python 3.x)

## Installation and Usage

To run the simulation:

```bash
python xy_vacuum_search.py [width] [height]
```

Substitute `[width]` and `[height]` with integers to define the grid size.

## Configuration

The turn cost can be enabled to simulate additional energy consumption for turning the vacuum cleaner. This feature affects the decision-making process of certain search algorithms like UCS and A*.

## Example Command

```bash
python xy_vacuum_search.py 10 10
```

This will start the simulation in a 10x10 grid.
