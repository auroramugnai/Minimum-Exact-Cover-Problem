from __future__ import annotations

import math
from typing import List, Dict, Optional, Tuple, Set, Union

import numpy as np
from qiskit import QuantumCircuit
# from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import Aer
from qiskit.circuit import Parameter, QuantumRegister
from qiskit.circuit.library import QAOAAnsatz, RXGate, MCMTVChain, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.visualization import plot_histogram

from utils_to_study_an_instance import *
from utils_for_plotting_and_reading import *

def get_parameters_from_user() -> Dict:
    """
    Prompts the user to input various parameters for the computation.

    This function asks the user to input values for the following parameters:
    - Number of layers (p)
    - Number of random attempts
    - Initialization string (init_string)
    - Size n
    - List of chosen instances
    - Choice for 'chosen_k'

    The function ensures that if no input is provided, default values are used.
    The list of chosen instances can be input as a comma-separated string, 
    either with square brackets or without.

    Returns
    -------
    dict
        A dictionary containing the parameters:
        - 'p' : int (number of layers)
        - 'random_attempts' : int (number of random attempts)
        - 'init_string' : str (initialization string)
        - 'n' : int (size)
        - 'chosen_instances' : List[int] (list of chosen instances)
        - 'chosen_k' : str (choice for 'chosen_k')
    """
    # Ask the user to input the parameters
    p = int(input("Number of layers (p), default is 3: ") or 3)
    random_attempts = int(input("Number of random attempts, default is 20: ") or 20)
    init_string = input("String initialization (all1 or all0), default is 'all1': ") or "all1"
    n = int(input("Size n (6, 8, 10), default is 6: ") or 6)

    # Ask for a list of numbers for the chosen instances
    chosen_instances = input("Number or list of numbers from 1 to 10 for chosen instances, default is [1, ..., 10]: ").strip()
    if chosen_instances:
        # Check if input is in the format "[2,3]" or "2,3"
        if '[' in chosen_instances and ']' in chosen_instances:
            # Parse input with square brackets
            chosen_instances = chosen_instances.strip("[]").split(',')
        else:
            # Parse input without square brackets
            chosen_instances = chosen_instances.split(',')
            
        # Convert to integers
        chosen_instances = [int(x.strip()) for x in chosen_instances]
    else:
        chosen_instances = range(1, 11)

    # Ask for the choice of k
    chosen_k = input("Choice for 'chosen_k' ('L=max(L_EC)', 'L=n', 'L=L_MEC'), default is 'L=L_MEC': ") or 'L=L_MEC'

    return {
            'p': p,
            'random_attempts': random_attempts,
            'init_string': init_string,
            'n': n,
            'chosen_instances': chosen_instances,
            'chosen_k': chosen_k
        }

def get_circuit_parameters(subsets: List[Set[int]], verbose: bool = False) -> Tuple[List[List[int]], int, int, int]:
    """
    Calculate and return circuit parameters based on the given subsets.

    Parameters
    ----------
    subsets : List[Set[int]]
        List of subsets representing the problem instance. Each subset corresponds 
        to a node in the graph.
    verbose : bool, optional, default=False
        If True, prints additional details during the execution.

    Returns
    -------
    list_of_intersections : List[List[int]]
        A list of intersections (connections) for each node of the instance. 
        Each element is a list of all the nodes that are connected to a given node.
    num_max_ctrl : int
        Maximum degree of the graph, representing the maximum number of intersections 
        (edges) for any node.
    NUM_ANC : int
        Number of ancillas required to implement the MCMTVChain for the circuit.
    QC_DIM : int
        Total number of qubits required for the circuit, which is the sum of the 
        number of nodes (subsets) and the number of ancillas.
    """
    # Build the graph for the subsets. Each node represents a subset, and intersections
    # indicate shared elements between subsets.
    list_of_intersections = build_instance_graph(subsets, verbose=False, draw_graph=False)
    
    # Calculate the maximum degree of the graph (num_max_ctrl).
    num_max_ctrl = max(len(neighbors) for neighbors in list_of_intersections)
    
    # Number of ancilla qubits required for the MCMTVChain implementation.
    NUM_ANC = num_max_ctrl - 1
    
    # Total qubits in the quantum circuit, including nodes and ancilla qubits.
    QC_DIM = len(subsets) + NUM_ANC

    if verbose:
        print("num_max_ctrl:", num_max_ctrl)
        print("NUM_ANC:", NUM_ANC)
        print("QC_DIM:", QC_DIM)

    return list_of_intersections, num_max_ctrl, NUM_ANC, QC_DIM



#############################################################################################################
#############################################################################################################
#############################################################################################################

def build_cost_circuit(n: int, instance: int, k: float, verbose: bool = False) -> Tuple[float, SparsePauliOp, QuantumCircuit]:
    """
    Build the cost Hamiltonian and its corresponding quantum circuit for the given problem instance.

    This function constructs the cost Hamiltonian as described in the paper by Wang et al..
    It also returns a quantum circuit that implements the Hamiltonian evolution.

    Parameters
    ----------
    n : int
        The dimension of the instance, i.e., the number of qubits used for the problem.
    instance : int
        The index or identifier of the specific instance being solved.
    k : float
        A parameter that defines the relationship between two components of the Hamiltonian.
    verbose : bool, optional, default=False
        If True, prints additional details about the construction of the circuit.

    Returns
    -------
    constant : float
        The constant term (-A + B) of the cost Hamiltonian, which does not affect the optimization process.
    hamiltonian : SparsePauliOp
        The cost Hamiltonian operator that encodes the problem.
    qc_ham : QuantumCircuit
        The quantum circuit that implements the Hamiltonian evolution, parametrized by gamma.
    """
    # Define the instance and its subsets based on the problem configuration.
    U, subsets_dict = define_instance(n, instance, verbose=verbose)
    subsets = list(subsets_dict.values())
    
    # Get quantum circuit parameters (e.g., number of qubits).
    _, _, _, QC_DIM = get_circuit_parameters(subsets, verbose=verbose)
    
    # Calculate lambda parameters based on instance size and k value.
    l2 = 1 / (n * len(U) - 2)
    l1 = k * n * l2  # Relationship: l1 / l2 = k * n

    # Compute terms of the Hamiltonian.
    A = l1 * sum(len(S) for S in subsets) / 2
    B = l2 * n / 2
    constant = -A + B  # Constant term of the Hamiltonian.
    
    # Create coefficients for Z operators.
    coeffs = [(l1 * len(S) / 2 - l2 / 2) for S in subsets]
    Z_operators = [("Z", [i], coeffs[i]) for i in range(n)]
    
    # Build the Hamiltonian as a SparsePauliOp.
    hamiltonian = SparsePauliOp.from_sparse_list(Z_operators, num_qubits=QC_DIM)
    
    # Print debug information if verbose mode is enabled.
    if verbose:
        print("A =", A)
        print("B =", B)
        print("constant = -A + B =", constant)
        print("\nhamiltonian:\n", hamiltonian)

    # Define the Hamiltonian evolution circuit using the parameter gamma.
    gamma = Parameter("gamma")  # Free parameter in the quantum circuit.
    evo = PauliEvolutionGate(hamiltonian, time=gamma)
    
    # Build the quantum circuit for the Hamiltonian evolution.
    qc_ham = QuantumCircuit(QC_DIM)
    qc_ham.append(evo, range(QC_DIM))
    
    # Decompose the circuit to simplify multi-qubit operations.
    qc_ham = qc_ham.decompose(reps=2)
    
    return constant, hamiltonian, qc_ham



#############################################################################################################
#############################################################################################################
#############################################################################################################

def build_mixing_circuit(n: int, instance: int, verbose: bool = False) -> QuantumCircuit:
    """
    Build the mixing Hamiltonian quantum circuit for the given problem instance.

    This function constructs a mixing operator for a QAOA-style quantum circuit, which
    explores the solution space by applying controlled rotations to the qubits. It returns
    a quantum circuit that implements the mixing evolution.

    Parameters
    ----------
    n : int
        The size of the problem instance, corresponding to the number of qubits used.
    instance : int
        The index or identifier of the specific instance being solved.
    verbose : bool, optional, default=False
        If True, prints additional information during the circuit construction.

    Returns
    -------
    qc_mixing : QuantumCircuit
        A quantum circuit implementing the mixing Hamiltonian evolution, parametrized by beta.
    """
    # Define the problem instance and extract subsets.
    U, subsets_dict = define_instance(n, instance, verbose=verbose)
    subsets = list(subsets_dict.values())
    
    # Extract circuit parameters: intersections, ancillas, and circuit dimensions.
    list_of_intersections, num_max_ctrl, NUM_ANC, QC_DIM = get_circuit_parameters(subsets, verbose=verbose)
    
    # Initialize quantum registers and quantum circuit.
    # The ancillas will be initialized to 1 in build_initialization_circuit.
    qr = QuantumRegister(n, 'q')
    qr_anc = QuantumRegister(NUM_ANC, 'ancilla')
    qc_mixing = QuantumCircuit(qr, qr_anc)

    # Define the parameter beta for X-rotations.
    beta = Parameter('beta')

    # Create a list of gates for controlled X-rotations using V-Chain (MCMTVChain).
    g = [MCMTVChain(RXGate(beta), x, 1) for x in range(1, num_max_ctrl + 1)]
    gates = [g[i].to_gate() for i in range(len(g))]

    # Add the gates to the quantum circuit, specifying the controls.
    # The order should be [controls, target, ancillas]. 
    # For example, with 5 qubits [0,1,2,3,4] and 2 ancillas [5,6], to apply an X rotation
    # on qubit 1 controlled by qubits [0,2,3], the gate will be added as:
    # qc_mixing.append(gates[2], [0,2,3, 1, 5,6])

    for i, intersections in enumerate(list_of_intersections):
        if intersections != []:
            N = len(intersections)
            qubits_list = intersections + [i] + list(range(n, n + N - 1))
            qc_mixing.append(gates[N - 1], qubits_list)

    # Print details if verbose mode is enabled.
    if verbose:
        print(f"Mixing circuit created for instance {instance} with {n} qubits.")
        print(f"Circuit dimension (QC_DIM): {QC_DIM}")
        print(f"Number of ancillas: {NUM_ANC}")
        print(f"Number of control gates: {num_max_ctrl}")

    return qc_mixing




#############################################################################################################
#############################################################################################################
############################################################################################################

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Multiple-Control, Multiple-Target Gate."""


from collections.abc import Callable

from qiskit import circuit
from qiskit.circuit import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError

from qiskit.circuit.library.standard_gates import XGate, YGate, ZGate, HGate, TGate, TdgGate, SGate, SdgGate


class MCMT(QuantumCircuit):
    """The multi-controlled multi-target gate, for an arbitrary singly controlled target gate.

    For example, the H gate controlled on 3 qubits and acting on 2 target qubit is represented as:

    .. parsed-literal::

        ───■────
           │
        ───■────
           │
        ───■────
        ┌──┴───┐
        ┤0     ├
        │  2-H │
        ┤1     ├
        └──────┘

    This default implementations requires no ancilla qubits, by broadcasting the target gate
    to the number of target qubits and using Qiskit's generic control routine to control the
    broadcasted target on the control qubits. If ancilla qubits are available, a more efficient
    variant using the so-called V-chain decomposition can be used. This is implemented in
    :class:`~qiskit.circuit.library.MCMTVChain`.
    """

    def __init__(
        self,
        gate: Gate | Callable[[QuantumCircuit, circuit.Qubit, circuit.Qubit], circuit.Instruction],
        num_ctrl_qubits: int,
        num_target_qubits: int,
    ) -> None:
        """Create a new multi-control multi-target gate.

        Args:
            gate: The gate to be applied controlled on the control qubits and applied to the target
                qubits. Can be either a Gate or a circuit method.
                If it is a callable, it will be casted to a Gate.
            num_ctrl_qubits: The number of control qubits.
            num_target_qubits: The number of target qubits.

        Raises:
            AttributeError: If the gate cannot be casted to a controlled gate.
            AttributeError: If the number of controls or targets is 0.
        """
        if num_ctrl_qubits == 0 or num_target_qubits == 0:
            raise AttributeError("Need at least one control and one target qubit.")

        # set the internal properties and determine the number of qubits
        self.gate = self._identify_gate(gate)
        self.num_ctrl_qubits = num_ctrl_qubits
        self.num_target_qubits = num_target_qubits
        self.ctrl_state = "0" * self.num_ctrl_qubits
        
        num_qubits = num_ctrl_qubits + num_target_qubits + self.num_ancilla_qubits

        # initialize the circuit object
        super().__init__(num_qubits, name="mcmt")
        self._label = f"{num_target_qubits}-{self.gate.name.capitalize()}"

        # build the circuit
        self._build()

    def _build(self):
        """Define the MCMT gate without ancillas."""
        if self.num_target_qubits == 1:
            # no broadcasting needed (makes for better circuit diagrams)
            broadcasted_gate = self.gate
        else:
            broadcasted = QuantumCircuit(self.num_target_qubits, name=self._label)
            for target in list(range(self.num_target_qubits)):
                broadcasted.append(self.gate, [target], [])
            broadcasted_gate = broadcasted.to_gate()

        mcmt_gate = broadcasted_gate.control(self.num_ctrl_qubits, ctrl_state= self.ctrl_state)
        self.append(mcmt_gate, self.qubits, [])

    @property
    def num_ancilla_qubits(self):
        """Return the number of ancillas."""
        return 0

    def _identify_gate(self, gate):
        """Case the gate input to a gate."""
        valid_gates = {
            "ch": HGate(),
            "cx": XGate(),
            "cy": YGate(),
            "cz": ZGate(),
            "h": HGate(),
            "s": SGate(),
            "sdg": SdgGate(),
            "x": XGate(),
            "y": YGate(),
            "z": ZGate(),
            "t": TGate(),
            "tdg": TdgGate(),
        }
        if isinstance(gate, ControlledGate):
            base_gate = gate.base_gate
        elif isinstance(gate, Gate):
            if gate.num_qubits != 1:
                raise AttributeError("Base gate must act on one qubit only.")
            base_gate = gate
        elif isinstance(gate, QuantumCircuit):
            if gate.num_qubits != 1:
                raise AttributeError(
                    "The circuit you specified as control gate can only have one qubit!"
                )
            base_gate = gate.to_gate()  # raises error if circuit contains non-unitary instructions
        else:
            if callable(gate):  # identify via name of the passed function
                name = gate.__name__
            elif isinstance(gate, str):
                name = gate
            else:
                raise AttributeError(f"Invalid gate specified: {gate}")
            base_gate = valid_gates[name]

        return base_gate

    def control(self, num_ctrl_qubits=1, label=None, annotated=False):
        ctrl_state = '0' * num_ctrl_qubits
        """Return the controlled version of the MCMT circuit."""
        if not annotated and ctrl_state is None:
            gate = MCMT(self.gate, self.num_ctrl_qubits + num_ctrl_qubits, self.num_target_qubits)
        else:
            gate = super().control(num_ctrl_qubits, label, ctrl_state, annotated=annotated)
            print(ctrl_state, num_ctrl_qubits, "ciao")
        return gate

    def inverse(self, annotated: bool = False):
        """Return the inverse MCMT circuit, which is itself."""
        return MCMT(self.gate, self.num_ctrl_qubits, self.num_target_qubits)


class MCMTVChain(MCMT):
    """The MCMT implementation using the CCX V-chain.

    This implementation requires ancillas but is decomposed into a much shallower circuit
    than the default implementation in :class:`~qiskit.circuit.library.MCMT`.

    **Expanded Circuit:**

    .. plot::

       from qiskit.circuit.library import MCMTVChain, ZGate
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = MCMTVChain(ZGate(), 2, 2)
       _generate_circuit_library_visualization(circuit.decompose())

    **Examples:**

        >>> from qiskit.circuit.library import HGate
        >>> MCMTVChain(HGate(), 3, 2).draw()

        q_0: ──■────────────────────────■──
               │                        │
        q_1: ──■────────────────────────■──
               │                        │
        q_2: ──┼────■──────────────■────┼──
               │    │  ┌───┐       │    │
        q_3: ──┼────┼──┤ H ├───────┼────┼──
               │    │  └─┬─┘┌───┐  │    │
        q_4: ──┼────┼────┼──┤ H ├──┼────┼──
             ┌─┴─┐  │    │  └─┬─┘  │  ┌─┴─┐
        q_5: ┤ X ├──■────┼────┼────■──┤ X ├
             └───┘┌─┴─┐  │    │  ┌─┴─┐└───┘
        q_6: ─────┤ X ├──■────■──┤ X ├─────
                  └───┘          └───┘
    """

    def _build(self):
        """Define the MCMT gate."""
        control_qubits = self.qubits[: self.num_ctrl_qubits]
        target_qubits = self.qubits[
            self.num_ctrl_qubits : self.num_ctrl_qubits + self.num_target_qubits
        ]
        ancilla_qubits = self.qubits[self.num_ctrl_qubits + self.num_target_qubits :]

        if len(ancilla_qubits) > 0:
            master_control = ancilla_qubits[-1]
        else:
            master_control = control_qubits[0]

        self._ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=False)
        for qubit in target_qubits:
            self.append(self.gate.control(ctrl_state='0'), [master_control, qubit], [])
        self._ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=True)

    @property
    def num_ancilla_qubits(self):
        """Return the number of ancilla qubits required."""
        return max(0, self.num_ctrl_qubits - 1)

    def _ccx_v_chain_rule(
        self,
        control_qubits: QuantumRegister | list[circuit.Qubit],
        ancilla_qubits: QuantumRegister | list[circuit.Qubit],
        reverse: bool = False        
    ) -> None:
        """Get the rule for the CCX V-chain.

        The CCX V-chain progressively computes the CCX of the control qubits and puts the final
        result in the last ancillary qubit.

        Args:
            control_qubits: The control qubits.
            ancilla_qubits: The ancilla qubits.
            reverse: If True, compute the chain down to the qubit. If False, compute upwards.

        Returns:
            The rule for the (reversed) CCX V-chain.

        Raises:
            QiskitError: If an insufficient number of ancilla qubits was provided.
        """
        
        ctrl_state='0'*len(control_qubits)
        
        if len(ancilla_qubits) == 0:
            return

        if len(ancilla_qubits) < len(control_qubits) - 1:
            raise QiskitError("Insufficient number of ancilla qubits.")

        iterations = list(enumerate(range(2, len(control_qubits))))
        if not reverse:
            self.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0], ctrl_state='00')
            for i, j in iterations:
                self.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1], ctrl_state='00')
        else:
            for i, j in reversed(iterations):
                self.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1], ctrl_state='00')
            self.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0], ctrl_state='00')

    def inverse(self, annotated: bool = False):
        return MCMTVChain(self.gate, self.num_ctrl_qubits, self.num_target_qubits)



#############################################################################################################
#############################################################################################################
#############################################################################################################

# #### Example to check that the 0-control works.

# In[10]:


# from qiskit import QuantumCircuit

# num_qubits = 10
# ctrl_qubit = 3

# qr = QuantumRegister(num_qubits, 'q')
# anc = QuantumRegister(8, 'ancilla')
# qc = QuantumCircuit(qr, anc)

# for i, bit in enumerate("0"*num_qubits):
#     qc.initialize(bit, i)

# theta = Parameter('theta') 
# gate = MCMTVChain(RXGate(theta), 9, 1).to_gate() 
# qc.append(gate, [0,1,2,4,5,6,7,8,9, ctrl_qubit, 10,11,12,13,14,15,16,17])
# qc.measure_all()
# qc.decompose().draw('mpl')

###########################################
# from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# from qiskit.primitives import StatevectorEstimator, StatevectorSampler
# import pandas as pd

# # Generate a pass manager without providing a backend
# pm = generate_preset_pass_manager(optimization_level=3)
# ansatz_isa = pm.run(qc)
# hamiltonian_isa = hamiltonian.apply_layout(ansatz_isa.layout)

# estimator = StatevectorEstimator()
# sampler = StatevectorSampler()

# qc = qc.assign_parameters([np.pi]) # ruoto di pi greco attorno a X
# qc_isa = pm.run(qc)
# result = sampler.run([qc_isa], shots=1024).result()
# samp_dist = result[0].data.meas.get_counts()
# samp_dist



#############################################################################################################
#############################################################################################################
#############################################################################################################

def build_initialization_circuit(
    n: int,
    instance: int,
    init_name: Union[str, List[str]],
    verbose: bool = False
) -> QuantumCircuit:
    """
    Constructs a quantum circuit for initialization with support for predefined or 
    custom qubit states.

    Parameters
    ----------
    n : int
        The number of qubits in the quantum circuit.
    instance : int
        An identifier for the problem instance, used to define specific subsets for the 
        circuit configuration.
    init_name : str or list of str
        Specifies the initialization state(s) for the qubits:
        - "all0": Initializes all qubits to the |0⟩ state.
        - "all1": Initializes all qubits to the |1⟩ state.
        - List of binary strings: Each string represents a desired state for the qubits 
        in superposition.
    verbose : bool, optional
        If True, enables verbose output for debugging purposes. Default is False.

    Returns
    -------
        QuantumCircuit
            A quantum circuit (`qc_initial`) initialized according to the specified configuration, 
            ready for further operations.
        dict
            A dictionary (`check_counts`) containing the measurement results and state counts, 
            useful for verifying the initialization. 
            This result can be plotted through `plot_histogram(check_counts)`
    """

    # Define instance-related parameters (e.g., problem-specific subsets)
    _, subsets_dict = define_instance(n, instance, verbose=verbose)
    subsets = list(subsets_dict.values())
    _, _, NUM_ANC, QC_DIM = get_circuit_parameters(subsets, verbose=verbose)

    # Initialize quantum registers and the quantum circuit
    qr = QuantumRegister(n, 'q')
    anc = QuantumRegister(NUM_ANC, 'ancilla')
    qc_initial = QuantumCircuit(qr, anc)

    # Initialize ancillas to |1⟩
    for ancilla in range(n, QC_DIM):
        qc_initial.initialize([0, 1], ancilla)  # |1⟩ state

    # Prepare initial states based on the `init_name` parameter
    if init_name == "all1":
        init_state = ["1" * n]  # Initialize all qubits to |1⟩
    elif init_name == "all0":
        init_state = ["0" * n]  # Initialize all qubits to |0⟩
    elif isinstance(init_name, list):
        init_state = init_name
    else:
        raise ValueError("Invalid `init_name`. Must be 'all0', 'all1',"
                         + "or a list of binary strings.")

    # Reverse binary strings for correct qubit order in Qiskit
    init_state = [state[::-1] for state in init_state]

    # Initialize state vector for selected states
    vec = np.zeros(2 ** n)
    for i in range(2 ** n):
        state = format(i, f'0{n}b')  # Generate binary state string
        if state in init_state:
            vec[i] = 1

    # Normalize the state vector
    vec = vec / np.linalg.norm(vec)

    # Initialize the quantum circuit with the normalized state vector
    state = Statevector(vec)
    qc_initial.initialize(state.data, list(range(n)))

    if verbose:
        print(f"Initialization circuit created for instance {instance} with {n} qubits.")
        print(f"Initial state(s): {init_state}")
        print(f"Quantum circuit dimension (QC_DIM): {QC_DIM}")
        print(f"Number of ancillas: {NUM_ANC}")

    # Check that everything went well.
    # Create an idependent copy
    qc_copy = qc_initial.copy()

    # Measure to check that the superposition was correct
    qc_copy.measure_all()

    # Simulate and visualize results
    svsim = Aer.get_backend('aer_simulator')
    qc_copy.save_statevector()
    result = svsim.run(qc_copy).result()
    check_counts = result.get_counts()
    
    return qc_initial, check_counts

#############################################################################################################
#############################################################################################################
#############################################################################################################

def cost_func(params, ansatz, hamiltonian, estimator):
    """
    Computes the energy estimate using the provided estimator.

    Parameters
    ----------
    params : ndarray
        Array of parameters for the ansatz circuit.
    ansatz : QuantumCircuit
        Parameterized quantum circuit (ansatz).
    hamiltonian : SparsePauliOp
        Hamiltonian operator for which the energy is estimated.
    estimator : Estimator
        Primitive for computing expectation values.

    Returns
    -------
    float
        Estimated energy value.
    """
    try:
        pub = (ansatz, [hamiltonian], [params])

        # Run the estimator to compute the expectation value
        result = estimator.run(pubs=[pub]).result()
        
        # Extract the energy value
        cost = result[0].data.evs[0]
        
        return cost

    except Exception as e:
        raise RuntimeError(f"Error in cost function computation: {e}")



#############################################################################################################
#############################################################################################################
#############################################################################################################

# def invert_counts(counts: Dict[str, int]) -> Dict[str, int]:
#     """
#     Reverses the bit order in the keys of a quantum measurement result dictionary.
    
#     This function is useful for correcting the bit order in measurement results, 
#     as quantum computing platforms might return results in little-endian order 
#     (least significant bit on the left), and reversing the bit order will make 
#     the string big-endian (most significant bit on the left).

#     Parameters
#     ----------
#     counts : dict
#         A dictionary where the keys are binary strings representing measurement 
#         outcomes, and the values are the counts (frequencies) of those outcomes.

#     Returns
#     -------
#     dict
#         A new dictionary with the keys reversed (bit order swapped).
    
#     Example
#     -------
#     If the input is:
#         {'00': 5, '01': 3, '10': 8, '11': 4}
    
#     The output will be:
#         {'00': 5, '10': 3, '01': 8, '11': 4}
    
#     The bit order in the keys is reversed in the returned dictionary.
#     """
#     # Reverse the bit order for each key and return the updated dictionary
#     return {k[::-1]: v for k, v in counts.items()}



#############################################################################################################
#############################################################################################################
#############################################################################################################

def find_gamma_bound(n: int, instance: int, k: float, verbose: bool = False) -> int:
    """
    Computes the upper bound for the gamma parameter in quantum optimization.

    The bound is derived by identifying the eigenvalue of the cost Hamiltonian with
    the smallest absolute value. This is useful for setting the maximum value of gamma 
    in quantum algorithms like QAOA (Quantum Approximate Optimization Algorithm).

    Parameters
    ----------
    n : int
        The dimension of the instance, typically the number of qubits involved in the problem.
    instance : int
        The identifier for the specific problem instance.
    k : float
        The scaling factor that influences the optimization.
    verbose : bool, optional
        If True, enables debug outputs to track the process. Default is False.

    Returns
    -------
    int
        The upper bound for gamma, rounded up to the nearest integer.

    Example
    -------
    If `n=10`, `instance=2`, and `k=0.5`, this function will compute the upper bound 
    for gamma based on the defined instance and the scaling factor `k`.
    """
    # Get the instance and its subsets from the problem definition
    U, subsets_dict = define_instance(n, instance, verbose=verbose)
    subsets = list(subsets_dict.values())

    # Scaling factor for l2, which is used in the calculation of the eigenvalue
    l2 = 1 / (n * len(U) - 2)

    # Example explanation:
    # For subsets like [[1, 2], [3], [4, 5, 6], [7, 8, 9, 10]], and a cumulative 
    # sum of subset lengths, we need to find the eigenvalue with the minimum 
    # absolute value based on the formula:
    # |f| = l2 * | Σx_i - k * n * Σw_i * x_i |

    # Sort subsets in increasing order of length.
    how_many_elements = lambda x: len(x)
    subsets_ord = sorted(subsets, key=how_many_elements)
    
    # Compute the cumulative sum of subset lengths. This helps in determining 
    # how much weight each subset contributes to the sum.
    cumul_subsets_len = [0]
    for i, s in enumerate(subsets_ord):
        cumul_subsets_len.append(cumul_subsets_len[-1] + len(s))
    
    # Calculate the minimum of f as described in the example.
    f_min = []
    for i, cumul in enumerate(cumul_subsets_len):
        f_min.append(l2 * (i - k * n * cumul))
    
    # Although the function might give values lower than -1, this is expected,
    # as we are dealing with subset lengths and not limited to a range of [-1, 1].
    
    # To find the maximum value for gamma, we identify the smallest absolute 
    # value from f_min (this will correspond to the value at position 1).
    gamma_max = 2 * np.pi / abs(f_min[1])
    
    # Round up to the nearest greater integer for gamma
    gamma_max = math.ceil(gamma_max)

    # Compute the gamma bound (half of the maximum gamma)
    gamma_bound = math.ceil(gamma_max / 2)

    # If verbose, print the computed gamma bounds for debugging purposes.
    if verbose:
        print(f"Gamma bounds -> [0, {2 * gamma_bound}] or [{-gamma_bound}, {gamma_bound}]")

    return gamma_bound



#############################################################################################################
#############################################################################################################
#############################################################################################################

# ## Example of constructing a QAOAAnsatz
# Uncomment the code if you wish to execute it.

# In[17]:


# n = 6  # Dimension of the problem (number of qubits)
# instance = 2  # Identifier of the chosen problem instance
# p = 1  # Number of layers for the QAOA (repetitions of the ansatz)
# h = 1  # Hamiltonian parameter (specific to the problem)

# ### Define the problem instance
# U, subsets_dict = define_instance(n, instance, verbose=False)  # Define the instance and get subsets
# subsets = list(subsets_dict.values())  # Extract subsets from the dictionary
# _, _, states_feasible, energies_feasible, EXACT_COVERS = find_spectrum(U, subsets_dict, n, h)  # Compute spectrum, get feasible states and energies

# # Identify MEC (Minimum Energy Cover) by selecting states with the minimum number of 1's in their binary representation
# MEC = [state for state in EXACT_COVERS if state.count("1") == min([x.count("1") for x in EXACT_COVERS])]

# ### Define the initial state for the QAOA
# # Generate binary strings of length n with exactly one '1' (all possible positions for the '1')
# one_one_states = ["".join(elem) for elem in distinct_permutations('0' * (n - 1) + '1')]
# init_name = one_one_states  # Set the initial state to be the list of states with exactly one '1'

# ### Prepare the cost and mixing circuits for the QAOA
# # Build the cost Hamiltonian circuit, which encodes the problem's cost function
# constant, hamiltonian, qc_cost = build_cost_circuit(n, instance, h, verbose=False)

# ### Build the mixing circuit, which is used to mix the states in the QAOA algorithm
# qc_mixing = build_mixing_circuit(n, instance, verbose=False)

# ### Build the initialization circuit with the initial state defined above
# qc_initial = build_initialization_circuit(n, instance, init_name, verbose=False, check=False)

# ### Put everything together to create the QAOA ansatz
# # The ansatz is a parameterized quantum circuit with the cost and mixing circuits, and the initial state
# ansatz = QAOAAnsatz(qc_cost, mixer_operator=qc_mixing, initial_state=qc_initial, reps=p, name='my_QAOA_circuit')

# # Disable LaTeX rendering globally in matplotlib
# import matplotlib
# matplotlib.rcParams['text.usetex'] = False
# matplotlib.rcParams['font.family'] = 'Dejavu Sans'

# # Plot the QAOA ansatz
# ansatz.decompose(reps=2).draw('mpl')

