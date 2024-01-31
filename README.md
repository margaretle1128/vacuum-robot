# Vacuum Cleaner Search Simulation

## Description

This project is a simulation of a vacuum cleaner agent operating in a 2D grid environment. The agent's goal is to clean all the dirty rooms while avoiding obstacles. The simulation includes various search algorithms to navigate the environment and clean it efficiently.

## Features

- Simulation of a vacuum cleaning agent in a grid environment.
- Support for different search algorithms:
  - Breadth-First Search (BFS)
  - Depth-First Search (DFS)
  - Uniform-Cost Search (UCS)
  - Greedy Best-First Search
  - A* Search
- Configurable grid size and environment through command-line arguments.
- GUI representation of the vacuum cleaner's environment and actions.

## Requirements

- Python 3.x
- Tkinter library (usually comes with Python)

## Installation

1. Ensure you have Python 3.x installed on your system.
2. Clone this repository or download the source code to your local machine.
3. No additional libraries are required outside of what is included with Python's standard library.

## Usage

To run the simulation with the default environment settings:

```bash
python xy_vacuum_search.py
```

To run the simulation with a custom environment size:

```bash
python xy_vacuum_search.py [width] [height]
```

Replace `[width]` and `[height]` with integers to set the size of the environment grid.

## GUI Controls

- **Reset:** Resets the environment to its initial state.
- **Next:** Executes the next step in the simulation.
- **Run:** Continuously runs the simulation without interruption.
- **TurnCost:** Toggles the cost for the vacuum cleaner to turn.
- **Search Type Dropdown:** Selects the search algorithm to be used by the vacuum cleaner.

## Example

```bash
python xy_vacuum_search.py 10 10
```

This command runs the simulation in a 10x10 grid environment.
