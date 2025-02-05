# Classical and Quantum Solvers for EC and MEC Problems 🔍💻

This repository contains Python code for solving EC (Exact Cover) and MEC (Minimum Exact Cover) problems, both using classical and quantum methods. The project includes several scripts that can generate random instances of these problems, solve them, and process the results. Below is a description of the key files in the repository. 

---
## Files Overview 📂

### 1. `random_EC_classical_solver.py` 🎲
This script allows you to generate random instances of the **Exact Cover (EC)** problem and solve them using a classical solver. It provides functionality to create problem instances and find solutions efficiently.

### 2. `random_MEC_classical_solver.py` 🎲
Similar to `random_EC_classical_solver.py`, this script generates random instances of the **Minimum Exact Cover (MEC)** problem and solves them using classical methods.

### 3. `EC_DWave.py` 🧑‍💻⚡
In this file, the **EC problem** is solved using quantum computing on the D-Wave platform. It utilizes instances defined in `instances.py`, solves them, and shows the results. Additionally, it saves output files in folders named using the current date and time (e.g., `<datetime>`).

### 4. `EC_DWave_postprocess.py` 🧐📊
This script processes the results generated by `EC_DWave.py`. It reads the files saved in date-named folders (e.g., `<datetime>`) and performs post-processing on the solutions to analyze and interpret the output.

### 5. `utils.py` 🔧
The `utils.py` file contains several utility functions that are helpful across the scripts. These functions are designed to assist with tasks like data manipulation, file handling, and other common operations needed in the EC and MEC problem-solving workflows.

---
## How to Use 📝

1. **Generate and solve EC/MEC problems**:
   - Run `random_EC_classical_solver.py` to generate and solve random instances of the EC problem.
   - Run `random_MEC_classical_solver.py` to generate and solve random instances of the MEC problem.

2. **Solve EC problem for fixed instances with D-Wave**:
   - Ensure you have the necessary credentials for D-Wave.
   - Run `EC_DWave.py` to solve an EC problem using the quantum computing capabilities of D-Wave, using instances from `instances.py`.

3. **Post-process D-Wave results**:
   - After running `EC_DWave.py`, use `EC_DWave_postprocess.py` to read and analyze the results saved in `<datetime>` folders.

4. **Use helper functions**:
   - Check `utils.py` for utility functions that can help you with various tasks throughout the project.

---
## Setting Up D-Wave LEAP API Token 🌐
Follow these simple steps to automatically set up your D-Wave API token for use in this project:

### Step 1: Install D-Wave Ocean SDK
First, install the D-Wave Ocean SDK with the following command:
```
pip install dwave-ocean-sdk
```
### Step 2: Configure Your API Token
Run this command to automatically configure your API token:
```
dwave configure --token <YOUR_API_TOKEN> --url https://cloud.dwavesys.com/sapi
```
Replace <YOUR_API_TOKEN> with the API token you received when signing up on D-Wave LEAP.
This command will save your token and D-Wave system preferences for future use.

### Step 3: Verify the Configuration
To make sure the token was set correctly, run:
```
dwave config
```
You should see your token and system information displayed.

### Step 4: Test the Connection
Check the connection to D-Wave by running:
```
dwave ping
```
If everything is set up correctly, you’ll see a success message.
