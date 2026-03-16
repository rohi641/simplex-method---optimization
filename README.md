# Two-Phase Simplex Solver (GitHub Version)

This folder contains a standalone implementation of the Two-Phase Simplex algorithm for solving Linear Programming (LP) problems.

## Features
- **Two-Phase Method**: Handles problems with any constraint type (`<=`, `>=`, `=`).
- **Variable Labeling**: Allows custom names for decision variables for easier interpretation of results.
- **Detailed Output**: Returns status, optimal objective value, and explicitly separates **Basic Variables (BV)** and **Non-Basic Variables (NBV)**.

## Folder Contents
- `two_phase.py`: The core solver implementation.
- `example.py`: A runnable script demonstrating the solver with a labeled example problem.
- `requirements.txt`: Python dependencies (NumPy).

## Installation

Ensure you have Python installed. It is recommended to use a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Usage

You can run the provided example to see the solver in action:

```bash
python example.py
```

### Example Code Snippet

```python
from two_phase import solve_two_phase
import numpy as np

c = np.array([2.0, 3.0, 4.0])
A = np.array([[-1.0, -2.0, -1.0], [2.0, 1.0, 3.0], [1.0, 1.0, 1.0]])
b = np.array([-10.0, 15.0, 6.0])
senses = [">=", "=", ">="]

# Custom labels
labels = ["Product_A", "Product_B", "Product_C"]

sol = solve_two_phase(c, A, b, senses, objective="max", variable_labels=labels)

print(f"Status: {sol['status']}")
print(f"Objective: {sol['objective']}")
print("BV:", sol["bv"])
print("NBV:", sol["nbv"])
```

## Requirements
- `numpy >= 1.21`
