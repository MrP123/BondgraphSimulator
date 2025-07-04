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
        # Quelle: e = konstante Anstrengung
        return [self.effort - self.value]

class Capacitor(BondGraphElement, StatefulElement):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def state_var(self) -> sp.Symbol:
        return sp.Symbol(f"q_{self.name}")
    
    @property
    def equations(self) -> list[sp.Expr]:
        # Dynamische Gleichung: f = dq/dt, e = q / C
        return [self.flow - sp.Derivative(self.state_var, 't'), self.effort - self.state_var / self.value]


class Inductor(BondGraphElement, StatefulElement):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def state_var(self) -> sp.Symbol:
        return sp.Symbol(f"p_{self.name}")
    
    @property
    def equations(self) -> list[sp.Expr]:
        # Dynamische Gleichung: e = dp/dt, f = p / L
        return [self.effort - sp.Derivative(self.state_var, 't'), self.flow - self.state_var / self.value]

class Resistor(BondGraphElement):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        # Widerstand: e = R * f
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

# Funktion zur Ableitung der Zustandsraumdarstellung
def derive_state_space(bond_graph):
    # Symbolische Variablen für Zustandsgrößen, Flüsse und Anstrengungen
    state_vars = {}  # Zustandsgrößen (z. B. q_C, p_L)
    efforts = {}     # Anstrengungen (e_C, e_L, ...)
    flows = {}       # Flüsse (f_C, f_L, ...)
    equations = []   # Liste der Gleichungen

    # Zustandsgrößen für C- und I-Elemente
    for element in bond_graph["elements"]:
        element: BondGraphElement
    
        if isinstance(element, StatefulElement):
            state_vars[element.name] = element.state_var

        efforts[element.name] = element.effort
        flows[element.name] = element.flow

        equations.extend(element.equations)

    # 0- und 1-Junctions
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

    # Symbolische Lösung der Gleichungen
    state_vars_list = list(state_vars.values())
    derivatives = [sp.Derivative(var, 't') for var in state_vars_list]
    solution = sp.solve(equations, derivatives + list(efforts.values()) + list(flows.values()))

    print(solution)

    # Matrizen A, B, C, D ableiten
    A = sp.zeros(len(state_vars_list), len(state_vars_list))
    B = sp.zeros(len(state_vars_list), 1)

    # Ableitungen der Zustandsgrößen (dx/dt) aus der Lösung extrahieren
    for i, var in enumerate(state_vars_list):
        derivative = sp.Derivative(var, 't')  # Symbolische Ableitung
        if derivative in solution:  # Prüfen, ob die Ableitung in der Lösung enthalten ist
            expr = solution[derivative]  # Ausdruck für dx/dt
            for j, state_var in enumerate(state_vars_list):
                A[i, j] = expr.diff(state_var)  # Partielle Ableitung nach Zustandsvariablen

            input_counter = 0
            for element in bond_graph["elements"]:
                element: BondGraphElement

                if isinstance(element, SourceEffort):
                    if element.effort in solution:
                        input_var = solution[element.effort]

                        B[i, input_counter] = expr.diff(input_var) # Partielle Ableitung nach Eingang (z. B. V)
                        input_counter += 1

            #input_var = solution[efforts["V"]]
            #B[i, 0] = expr.diff(input_var)  

    for element in bond_graph["elements"]:
        
        if isinstance(element, Capacitor):
            if element.effort in solution:
                C_voltage_C = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_voltage_C[0, j] = solution[element.effort].diff(state_var)
                
            if element.flow in solution:
                C_current_C = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_current_C[0, j] = solution[element.flow].diff(state_var)

        elif isinstance(element, Inductor):
            if element.effort in solution:
                C_voltage_I = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_voltage_I[0, j] = solution[element.effort].diff(state_var)
                
            if element.flow in solution:
                C_current_I = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_current_I[0, j] = solution[element.flow].diff(state_var)

        elif isinstance(element, Resistor):
            if element.effort in solution:
                C_voltage_R = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_voltage_R[0, j] = solution[element.effort].diff(state_var)
                
            if element.flow in solution:
                C_current_R = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_current_R[0, j] = solution[element.flow].diff(state_var)

    # Matrix C und D
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

# Zustandsraumdarstellung ableiten
A, B, C_mats, D = derive_state_space(bond_graph)

# Ergebnisse anzeigen
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


x0 = sp.MatrixSymbol("x0", A.shape[0], 1)  # Anfangszustand als Matrix

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

plt.plot(T, yout)
plt.title("Step Response of the Bond Graph System")
plt.show()