from enum import Enum
from abc import ABC, abstractmethod

import sympy as sp

class StatefulElement(ABC):
    @property
    @abstractmethod
    def state_var(self) -> sp.Symbol:
        raise NotImplementedError("Subclasses should implement this method")

class BondgraphObject(ABC):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

class BondGraphElement(BondgraphObject, ABC):
    def __init__(self, name: str, value: str):
        super().__init__(name)
        self.value = sp.Symbol(value)
    
    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        raise NotImplementedError("Subclasses should implement this method")
    
    @property
    def effort(self) -> sp.Symbol:
        return sp.Symbol(f"e_{self.name}") 
    
    @property
    def flow(self) -> sp.Symbol:
        return sp.Symbol(f"f_{self.name}") 

    def __repr__(self):
        return f"{self.type}({self.name}, {self.value})"

class SourceEffort(BondGraphElement):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        # Source --> constant effort
        return [self.effort - self.value]

class Capacitor(BondGraphElement, StatefulElement):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def state_var(self) -> sp.Symbol:
        return sp.Symbol(f"q_{self.name}")
    
    @property
    def equations(self) -> list[sp.Expr]:
        # Dynamic equation: f = dq/dt, e = q / C
        return [self.flow - sp.Derivative(self.state_var, 't'), self.effort - self.state_var / self.value]


class Inductor(BondGraphElement, StatefulElement):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def state_var(self) -> sp.Symbol:
        return sp.Symbol(f"p_{self.name}")
    
    @property
    def equations(self) -> list[sp.Expr]:
        # Dynamic equation: e = dp/dt, f = p / L
        return [self.effort - sp.Derivative(self.state_var, 't'), self.flow - self.state_var / self.value]

class Resistor(BondGraphElement):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        # Resistance: e = R * f
        return [self.effort - self.value * self.flow]

class Junction(BondgraphObject, ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self.flows: list[sp.Symbol] = []
        self.efforts: list[sp.Symbol] = []

    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        raise NotImplementedError("Subclasses should implement this method")

class OneJunction(Junction):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def equations(self) -> list[sp.Expr]:
        return [sum(self.efforts), *[self.flows[0] - f for f in self.flows[1:]]]
    
class ZeroJunction(Junction):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def equations(self) -> list[sp.Expr]:
        return [sum(self.flows), *[self.efforts[0] - e for e in self.efforts[1:]]]

class Bond:
    def __init__(self, from_element: BondgraphObject, to_element: BondgraphObject):
        self.from_element = from_element
        self.to_element = to_element

# Beispiel: Bondgraph-Datenstruktur
voltage_source = SourceEffort("V", "V(t)")  # Spannungsquelle
capacitor = Capacitor("C", "C")             # Kapazität
inductor = Inductor("L", "L")               # Induktivität
resistor = Resistor("R", "R")               # Widerstand
junction = OneJunction("J")                 # 1-Junction

bond_graph = {
    "elements": [
        voltage_source,
        capacitor,
        inductor,
        resistor,
    ],
    "connections": [
        Bond(voltage_source, junction),  # Spannungsquelle an 1-Junction
        Bond(junction, capacitor),       # 1-Junction an Kondensator
        Bond(junction, resistor),        # 1-Junction an Widerstand
        Bond(junction, inductor),        # 1-Junction an Induktivität
    ]
}

import sympy as sp

# Calculate state space representation of linear bond graph
def derive_state_space(bond_graph):
    # symbolic vars for states, efforts and flows
    state_vars = {}  # states (z. B. q_C, p_L)
    efforts = {}     # (e_C, e_L, ...)
    flows = {}       # (f_C, f_L, ...)
    equations = []   # list of all equations

    # Add all state variables, efforts and flows from elements
    for element in bond_graph["elements"]:
        element: BondGraphElement
    
        if isinstance(element, StatefulElement):
            state_vars[element.name] = element.state_var

        efforts[element.name] = element.effort
        flows[element.name] = element.flow

        equations.extend(element.equations)

    # Go through all connected bonds and collect junctions
    # Append the efforts and flows of the elements connected to each junction
    for bond in bond_graph["connections"]:
        bond: Bond

        if isinstance(bond.from_element, Junction) or isinstance(bond.to_element, Junction):
            junction = bond.from_element if isinstance(bond.from_element, Junction) else bond.to_element
            element = bond.to_element if junction is bond.from_element else bond.from_element
            
            junction.flows.append(element.flow)
            junction.efforts.append(element.effort)

    # After all junctions have been processed, collect their equations
    for bond in bond_graph["connections"]:
        bond: Bond

        if isinstance(bond.from_element, Junction):
            equations.extend(bond.from_element.equations)
        elif isinstance(bond.to_element, Junction):
            equations.extend(bond.to_element.equations)

    # Symbolically solve system of equations
    state_vars_list = list(state_vars.values())
    state_derivatives = [sp.Derivative(var, 't') for var in state_vars_list]
    solution = sp.solve(equations, state_derivatives + list(efforts.values()) + list(flows.values()))

    print(solution)

    # Determine A, B, C, D matrices for SS representation
    n_states = len(state_vars_list)
    n_inputs = 1
    n_outputs = 1

    A = sp.zeros(n_states, n_states)
    B = sp.zeros(n_states, n_inputs)

    # Extract the state derivatives (dx/dt) from the solution
    for i, var in enumerate(state_vars_list):
        derivative = sp.Derivative(var, 't')  # symbolic derivatives
        if derivative in solution:  # check if they are in the solution
            expr = solution[derivative]  # expression for dx/dt
            for j, state_var in enumerate(state_vars_list):
                A[i, j] = expr.diff(state_var)  # partial derivative w.r.t. state variables as A matrix links dx/dt with x via dx/dt ~ A*x

            input_counter = 0
            for element in bond_graph["elements"]:
                element: BondGraphElement

                # Only check source elements for inputs
                if isinstance(element, SourceEffort):
                    if element.effort in solution:
                        input_var = solution[element.effort]

                        B[i, input_counter] = expr.diff(input_var) # partial derivatives of dx/dt w.r.t. inputs as dx/dt ~ B*u
                        input_counter += 1

            #input_var = solution[efforts["V"]]
            #B[i, 0] = expr.diff(input_var)  

    for element in bond_graph["elements"]:
        # Only check non-source elements
        if not isinstance(element, SourceEffort):

            # Get output matrices C for effort and flow in element
            if element.effort in solution:
                C_effort = sp.zeros(n_outputs, n_states)
                for j, state_var in enumerate(state_vars_list):
                    C_effort[0, j] = solution[element.effort].diff(state_var)
                
            if element.flow in solution:
                C_flow = sp.zeros(n_outputs, n_states)
                for j, state_var in enumerate(state_vars_list):
                    C_flow[0, j] = solution[element.flow].diff(state_var)

            # Assign to specific variables based on element type
            if isinstance(element, Capacitor):
                C_voltage_C = C_effort
                C_current_C = C_flow
            elif isinstance(element, Inductor):
                C_voltage_I = C_effort
                C_current_I = C_flow
            elif isinstance(element, Resistor):
                C_voltage_R = C_effort
                C_current_R = C_flow

    # All variants of the C matrix
    C_mats = {
        "voltage_C": C_voltage_C,
        "current_C": C_current_C,
        "voltage_I": C_voltage_I,
        "current_I": C_current_I,
        "voltage_R": C_voltage_R,
        "current_R": C_current_R,
    }

    D = sp.zeros(1, 1)

    return A, B, C_mats, D

# Calculate state space representation
A, B, C_mats, D = derive_state_space(bond_graph)

# Print results
print("Matrix A:")
sp.pprint(A)
print("\nMatrix B:")
sp.pprint(B)
print("\nMatrix C:")
for key, C_l in C_mats.items():
    print(f"\n{key}:")
    sp.pprint(C_l)
print("\nMatrix D:")
sp.pprint(D)


x0 = sp.MatrixSymbol("x0", A.shape[0], 1)  # IC as matrix symbol

init_cond = sp.Eq(sp.Symbol("Uc"), (C_mats["voltage_C"] * x0)[0])
print("\nInitial condition:")
sp.pprint(init_cond)

init_cond_solution = sp.solve(init_cond, x0[0, 0])
print("\nSolution for initial condition:")
sp.pprint(init_cond_solution)


import numpy as np

R_val = 470
C_val = 10e-6
L_val = 1e-3
V_val = 5
Uc_0_val = 5

subs_dict = {
    "R": R_val,
    "C": C_val,
    "L": L_val,
    "V(t)": V_val
}

A_mat_val = np.array(A.subs(subs_dict), dtype=np.float64)
B_mat_val = np.array(B.subs(subs_dict), dtype=np.float64)
C_mat_val = {key: np.array(val.subs(subs_dict), dtype=np.float64) for key, val in C_mats.items()}
D_mat_val = np.array(D.subs(subs_dict), dtype=np.float64)

x0_val = np.zeros_like(B_mat_val)
#x0_val[0, 0] = init_cond_solution[0].subs({"Uc": Uc_0_val, "C": C_val})
#x0_val = np.array(x0_val, dtype=np.float64)

import control as ctrl
import matplotlib.pyplot as plt

print(f"Running simulation with\n A:\n{A_mat_val}\nB:\n{B_mat_val}\nC:\n{C_mat_val}\nD:\n{D_mat_val}\nx0:\n{x0_val}")

sys = ctrl.ss(A_mat_val, -B_mat_val, C_mat_val["voltage_C"], -D_mat_val)
T, yout = ctrl.step_response(sys, T=50.0e-3, X0=x0_val)

print(f"Max time step: {np.max(np.diff(T))}")

plt.plot(T, yout)
plt.title("Step Response of the Bond Graph System")
plt.show()