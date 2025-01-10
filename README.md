# Minimum-Exact-Cover-Problem and Quantum Algorithms

This repository contains two main directories, **ECP_c-and-q** and **MECP_with-QAOA**, dedicated to solving the Exact Cover Problem and Minimum Exact Cover Problem using classical and quantum computing approaches.

## Directory Structure

### 📁 **MECP_with-QAOA**
This directory is dedicated to:
1. Implementing the **QAOA+ (Quantum Alternating Operator Ansatz)** on Qiskit to solve the Minimum Exact Cover Problem (MECP).
2. Solving the Minimum Exact Cover Problem on fixed instances using QAOA+.

### 📁 **ECP_c-and-q**
This directory focuses on:
1. **Solving random instances** of the Exact Cover Problem (ECP) and Minimum Exact Cover Problem (MECP) using classical strategies (c).
2. **Solving fixed instances** of the Exact Cover Problem using the D-Wave quantum annealer (q).


---

## 📚 Accessing `.ipynb` Files with JupyterLab or Jupyter Notebook

This repository contains Jupyter Notebook (`.ipynb`) files that you can open and modify using **JupyterLab** or **Jupyter Notebook**. Follow the instructions below to set up your environment and access the files.

### 🛠️ Prerequisites
Ensure you have the following tools installed on your system:
- **Python** (version 3.7 or higher)
- **pip**
- **JupyterLab** or **Jupyter Notebook**

If you don't have these installed, run the following command in your terminal to install both:
```
pip install jupyterlab notebook
```

### 📥 Clone the Repository
To access the files, you'll need to clone this repository to your computer. Run one of the following commands in your terminal:
- HTTPS
  ```
  git clone https://github.com/auroramugnai/Minimum-Exact-Cover-Problem.git
  ```

- SSH
  ```
  git clone git@github.com:auroramugnai/Minimum-Exact-Cover-Problem.git
  ```

### 🚀 Launch JupyterLab or Jupyter Notebook
Navigate to the repository directory:
```
cd Minimum-Exact-Cover-Problem
```

#### Option 1: Launch Jupyter Notebook
Start Jupyter Notebook by running:
```
jupyter notebook
```
This will open your default browser with the Jupyter Notebook interface. If it doesn't, copy and paste the link provided in the terminal into your browser's address bar.

#### Option 2: Launch JupyterLab
Start JupyterLab by running:
```
jupyter lab
```
This will open your default browser with the JupyterLab interface. Similarly, if it doesn't open automatically, copy and paste the link from the terminal into your browser.

### 📂 Open the `.ipynb` Files
In the JupyterLab or Jupyter Notebook interface, use the file browser to navigate to the repository directory.  
Click on a `.ipynb` file to open it and start working.


---

## 📖 Accessing Function Documentation with `help()`
Using `help()` is useful when you're working with unfamiliar functions or libraries. To access the documentation of a function `my_function` , simply call:
```
help(my_function)
```
This will display the docstring of `my_function`, which will typically contain:

- A description of what the function does.
- The parameters the function accepts, including their types and default values.
- The return values of the function.
- Any additional notes on how the function should be used.



