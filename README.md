# CGPoisson

## Introduction

This repository is the final project for the AsPhys8030 course at National Taiwan University (NTU). It showcases various computational methods for solving Poisson equations. The primary components include:

- A C++ implementation using the Conjugate Gradient method.
- A Python script that leverages PyTorch to perform computations on both CPU and GPU.
- A Jupyter Notebook that provides a detailed demonstration of solving Poisson equations.

Each component is designed to offer a unique approach to solving Poisson equations, highlighting the versatility and efficiency of different computational techniques.

## Repository Contents

### 1. `CGPoisson.cpp`
This file contains a C++ implementation of solving the Poisson equation using the Conjugate Gradient method. The Poisson equation is a fundamental partial differential equation in mathematical physics.

- **Key Features:**
  - Implementation of the Conjugate Gradient method
  - Efficient handling of large sparse matrices
  - Example usage of the solver
  
- **How to Compile:**
  ```sh
  g++ -o CGPoisson CGPoisson.cpp
  ```

- **How to Run:**
  ```sh
  ./CGPoisson
  ```

### 2. `demo_pytorch.ipynb`
This Jupyter Notebook demonstrates the use of PyTorch, a popular deep learning library, for solving Poisson equations. It includes detailed explanations and visualizations to help understand the process.

- **Key Features:**
  - Step-by-step guide to solving Poisson equations with PyTorch
  - Visualizations of the solutions
  - Interactive cells to experiment with different parameters

- **How to Use:**
  Open the notebook in Jupyter:
  ```sh
  jupyter notebook demo_pytorch.ipynb
  ```

### 3. `poisson.py`
This Python script provides an implementation of solving Poisson equations using numerical methods. It can be run as a standalone script and includes comments for better understanding.

- **Key Features:**
  - Python implementation of numerical methods for solving Poisson equations
  - Easy to modify and extend for different use cases
  - Includes test cases and example runs

- **How to Run:**
  ```sh
  python poisson.py
  ```

## Getting Started

To get started with this repository, clone it to your local machine:

```sh
git clone https://github.com/yourusername/your-repo-name.git
```

Ensure you have the necessary dependencies installed. For the Python components, you can create a virtual environment and install the required packages:

```sh
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

For the C++ component, ensure you have a C++ compiler like `g++`.

## Usage

### Running the C++ Solver
1. Compile the `CGPoisson.cpp` file.
2. Execute the compiled program.

### Using the PyTorch Notebook
1. Launch Jupyter Notebook.
2. Open `demo_pytorch.ipynb` and follow the instructions within.

### Running the Python Script
1. Ensure all dependencies are installed.
2. Run the `poisson.py` script.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests to contribute. Ensure your code follows the repository's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

