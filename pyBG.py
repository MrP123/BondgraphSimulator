from abc import ABC, abstractmethod

import sympy as sp

class StatefulElement(ABC):
    @property
    @abstractmethod
    def state_var(self) -> sp.Symbol:
        raise NotImplementedError("Subclasses should implement this method")


class Node(ABC):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"
    
class Bond:
    counter = 0

    def __init__(self, from_element: Node, to_element: Node, causality: str):
        self.from_element = from_element
        self.to_element = to_element
        self.causality = causality

        if causality not in ["effort_out", "flow_out"]:
            raise ValueError(f"Invalid causality: {causality}. Must be 'effort_out' or 'flow_out'.")
        
        self.num = Bond.counter
        Bond.counter += 1

        self.effort = sp.Symbol(f"e_{self.num}")
        self.flow = sp.Symbol(f"f_{self.num}")

    def __repr__(self):
        return f"Bond(from={self.from_element}, to={self.to_element}, causality={self.causality})"

class ElementOnePort(Node, ABC):
    def __init__(self, name: str, value: str):
        super().__init__(name)
        self.value = sp.Symbol(value, real=True, positive=True)  # Ensure value is a positive real number
        self.bond: Bond = None # bond that connects this element to a bond graph
    
    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        raise NotImplementedError("Subclasses should implement this method")

    def __repr__(self):
        return f"{self.__class__.__name__}(name = {self.name}, value = {self.value})"

class SourceEffort(ElementOnePort):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        # Source --> constant effort
        return [self.bond.effort - self.value]

class SourceFlow(ElementOnePort):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        # Source --> constant flow
        return [self.bond.flow - self.value]

class Capacitor(ElementOnePort, StatefulElement):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def state_var(self) -> sp.Symbol:
        return sp.Symbol(f"q_{self.name}")
    
    @property
    def equations(self) -> list[sp.Expr]:
        # Dynamic equation: f = dq/dt, e = q / C
        return [self.bond.flow - sp.Derivative(self.state_var, 't'), self.bond.effort - self.state_var / self.value]

class Inductor(ElementOnePort, StatefulElement):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def state_var(self) -> sp.Symbol:
        return sp.Symbol(f"p_{self.name}")
    
    @property
    def equations(self) -> list[sp.Expr]:
        # Dynamic equation: e = dp/dt, f = p / L
        return [self.bond.effort - sp.Derivative(self.state_var, 't'), self.bond.flow - self.state_var / self.value]

class Resistor(ElementOnePort):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        # Resistance: e = R * f
        if self.bond.causality == "effort_out":
            return [self.bond.flow - 1/self.value * self.bond.effort]
        elif self.bond.causality == "flow_out":
            return [self.bond.effort - self.value * self.bond.flow]
    
class ElementTwoPort(Node, ABC):
    def __init__(self, name: str, value: str):
        super().__init__(name)
        self.value = sp.Symbol(value, real=True, positive=True)
        self.bond1: Bond = None # ElementOther   --(bond1)--> ElementTwoPort
        self.bond2: Bond = None # ElementTwoPort --(bond2)--> ElementOther
    
    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        raise NotImplementedError("Subclasses should implement this method")

    def __repr__(self):
        return f"{self.__class__.__name__}(name = {self.name}, value = {self.value})"

class Transformer(ElementTwoPort):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        if self.bond1 is None or self.bond2 is None:
            raise ValueError("Both bonds must be assigned to the transformer element.")

        if self.bond1.causality != self.bond2.causality:
            raise ValueError("Both bonds must have the same causality for a transformer element.")

        if self.bond1.causality == "effort_out":
            return [self.bond1.flow - 1/self.value * self.bond2.flow, self.bond2.effort - 1 / self.value * self.bond1.effort]
        elif self.bond1.causality == "flow_out":
            return [self.bond1.effort - self.value * self.bond2.effort, self.bond2.flow - self.value * self.bond1.flow]
        
class Gyrator(ElementTwoPort):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        if self.bond1 is None or self.bond2 is None:
            raise ValueError("Both bonds must be assigned to the gyrator element.")

        if self.bond1.causality == self.bond2.causality:
            raise ValueError("Both bonds must have the different causality for a gyrator element.")

        if self.bond1.causality == "effort_out":
            return [self.bond1.flow - 1/self.value * self.bond2.effort, self.bond2.flow - 1 / self.value * self.bond1.effort]
        elif self.bond1.causality == "flow_out":
            return [self.bond1.effort - self.value * self.bond2.flow, self.bond2.effort - self.value * self.bond1.flow]
        

class Junction(Node, ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self.bonds: list[Bond] = []
        self.strong_bond = None

    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        raise NotImplementedError("Subclasses should implement this method")

class OneJunction(Junction):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def equations(self) -> list[sp.Expr]:
        effort_eq = 0
        for b in self.bonds:
            dir = +1 if b.to_element == self else -1
            effort_eq += dir * b.effort

        flow_eq = [self.bonds[0].flow - b.flow for b in self.bonds[1:]]
        return [effort_eq, *flow_eq]
    
class ZeroJunction(Junction):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def equations(self) -> list[sp.Expr]:
        flow_eq = 0
        for b in self.bonds:
            dir = +1 if b.to_element == self else -1
            flow_eq += dir * b.flow

        effort_eq = [self.bonds[0].effort - b.effort for b in self.bonds[1:]]
        return [flow_eq, *effort_eq]

class BondGraph():

    def __init__(self):
        self.elements = []
        self.connections: list[Bond] = []

        self.state_vars: list[sp.Expr] = []
        self.equations: list[sp.Expr] = []
        self.inputs: list[sp.Symbol] = []

    def add_element(self, element: Node):
        self.elements.append(element)

        if isinstance(element, SourceEffort) or isinstance(element, SourceFlow):
            self.inputs.append(element.value)

    def add_connection(self, bond: Bond):
        self.connections.append(bond)

    def handle_bonds(self):
        
        def handle_bond_element(element: Node, bond: Bond):
            if isinstance(element, ElementOnePort):
                element.bond = bond
                
            elif isinstance(element, ElementTwoPort):
                if bond.to_element == element:
                    element.bond1 = bond
                elif bond.from_element == element:
                    element.bond2 = bond

            elif isinstance(element, Junction):
                element.bonds.append(bond)

                if isinstance(element, OneJunction):
                    if (bond.from_element == element and bond.causality == "effort_out") or (bond.to_element == element and bond.causality == "flow_out"):
                        # bond is strong bond for zero junction
                        if element.strong_bond is None:
                            element.strong_bond = bond
                            print(f"Assigned strong bond {bond} to OneJunction {element}.")
                        else:
                            raise ValueError(f"OneJunction {element} already has a strong bond: {element.strong_bond}. Cannot assign {bond}.")

                elif isinstance(element, ZeroJunction):
                    if (bond.from_element == element and bond.causality == "flow_out") or (bond.to_element == element and bond.causality == "effort_out"):
                        # bond is strong bond for zero junction
                        if element.strong_bond is None:
                            element.strong_bond = bond
                            print(f"Assigned strong bond {bond} to ZeroJunction {element}.")
                        else:
                            raise ValueError(f"ZeroJunction {element} already has a strong bond: {element.strong_bond}. Cannot assign {bond}.")
        
        for bond in self.connections:
            handle_bond_element(bond.from_element, bond)
            handle_bond_element(bond.to_element, bond)

    def handle_equations(self):
        for element in self.elements:
            
            if isinstance(element, ElementOnePort):
                bond = element.bond
                if bond is None:
                    raise ValueError(f"Element {element} has no connected bond.")
                
                # Add equations from the element to the bond graph
                bond_graph.equations.extend(element.equations)

                if isinstance(element, StatefulElement):
                    bond_graph.state_vars.append(element.state_var)

            elif isinstance(element, ElementTwoPort):
                bond1 = element.bond1
                bond2 = element.bond2
                if bond1 is None or bond2 is None:
                    raise ValueError(f"Element {element} has no connected bonds.")
                
                # Add equations from the element to the bond graph
                bond_graph.equations.extend(element.equations)

            elif isinstance(element, Junction):
                bond_graph.equations.extend(element.equations)

    def get_solution_equations(self):
        state_derivatives = [sp.Derivative(var, 't') for var in self.state_vars]

        solution = sp.solve(self.equations, state_derivatives + [b.effort for b in self.connections] + [b.flow for b in self.connections])
        return solution

bond_graph = BondGraph()

#https://en.wikipedia.org/wiki/Bond_graph#:~:text=denotes%20preferred%20causality.-,For%20the%20example%20provided,-%2C
voltage_source = SourceEffort("V", "V(t)")
capacitor = Capacitor("C6", "C6")
inductor = Inductor("I3", "L3")
resistor2 = Resistor("R2", "R2")
resistor7 = Resistor("R7", "R7")
transformer = Transformer("T1", "T1")
junction1 = OneJunction("J1")
junction0 = ZeroJunction("J0")

bond_graph.add_element(voltage_source)
bond_graph.add_element(capacitor)
bond_graph.add_element(inductor)
bond_graph.add_element(resistor2)
bond_graph.add_element(resistor7)
bond_graph.add_element(transformer)
bond_graph.add_element(junction1)
bond_graph.add_element(junction0)

bond_graph.add_connection(Bond(voltage_source, junction1, "effort_out"))
bond_graph.add_connection(Bond(junction1, resistor2, "flow_out"))
bond_graph.add_connection(Bond(junction1, inductor, "effort_out"))
bond_graph.add_connection(Bond(junction1, transformer, "flow_out"))
bond_graph.add_connection(Bond(transformer, junction0, "flow_out"))
bond_graph.add_connection(Bond(junction0, capacitor, "flow_out"))
bond_graph.add_connection(Bond(junction0, resistor7, "effort_out"))

#bond_graph.add_element(voltage_source)
#bond_graph.add_element(capacitor)
#bond_graph.add_element(inductor)
#bond_graph.add_element(resistor)
#bond_graph.add_element(junction)

# Create bonds between elements and junction
#bond_graph.add_connection(Bond(voltage_source, junction, "effort_out"))
#bond_graph.add_connection(Bond(junction, capacitor, "flow_out"))
#bond_graph.add_connection(Bond(junction, resistor, "flow_out"))
#bond_graph.add_connection(Bond(junction, inductor, "effort_out"))

bond_graph.handle_bonds()
bond_graph.handle_equations()
solution = bond_graph.get_solution_equations()

n_states = len(bond_graph.state_vars)
n_inputs = len(bond_graph.inputs)  # Number of inputs (sources)
n_outputs = 2*len(bond_graph.connections) #effort & flow for each bond

# General form of a state space model
# x_dot = f(x, u)
#     y = h(x, u)
# Simplification for linear systems:
# x_dot = A*x + B*u
#     y = C*x + D*u
# --> therefore
# A = ∂f/∂x, B = ∂f/∂u, C = ∂h/∂x, D = ∂h/∂u each at stationary point 0

f: sp.Matrix = sp.zeros(n_states, 1)
for i, state_var in enumerate(bond_graph.state_vars):
    state_deriv = sp.Derivative(state_var, 't')  # symbolic derivative dx/dt
    f[i] = solution[state_deriv]

h: sp.Matrix = sp.zeros(n_outputs, 1) # efforts then flows
for i, bond in enumerate(bond_graph.connections):
    h[i] = solution[bond.effort]
    h[i + n_outputs//2] = solution[bond.flow]

A = f.jacobian(bond_graph.state_vars)
B = f.jacobian(bond_graph.inputs)

C = h.jacobian(bond_graph.state_vars)
D = h.jacobian(bond_graph.inputs)
# alternatively could use sp.linear_eq_to_matrix(...)

# Print results
print("Matrix A:")
sp.pprint(A)
print("\nMatrix B:")
sp.pprint(B)
print("\nMatrix C:")
sp.pprint(C)
print("\nMatrix D:")
sp.pprint(D)

# Initial conditions for the state variables --> not yet working
#x0_entries = sp.symbols(f"x0_0:{n_states}", real=True)
#y0_entries = sp.symbols(f"y0_0:{n_outputs}", real=True)
#x0 = sp.Matrix(x0_entries)
#y0 = sp.Matrix(y0_entries)
#U0_C6 = sp.symbols('U0_C6', real=True)
#U0_I3 = sp.symbols('U0_I3', real=True)

#free_params = sp.symbols('y0_0:2 y0_3:5 y0_6:14', real=True)
#y0_free = sp.Matrix([*free_params[:2], U0_I3, *free_params[2:5], U0_C6, *free_params[5:]])

#for i, bond in enumerate(bond_graph.connections):
#    if bond.effort == capacitor.bond.effort:
#        print(f"capacitor = {i}")
#    if bond.flow == inductor.bond.flow:
#        print(f"inductor = {i}")

#equation = C * x0 - y0_free

#scalar_equations = [equation[i, 0] for i in range(n_outputs)]

# Solve the system for x0
#solutions = sp.linsolve(scalar_equations, x0)



import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

R_val = 470.0
C_val = 10e-6
L_val = 1e-3
V_val = 1.0
T_val = 2.0

#ToDo: maybe make numeric value part of the element class?
subs_dict = {
    resistor2.value:      R_val,
    resistor7.value:      R_val,
    capacitor.value:      C_val,
    inductor.value:       L_val,
    transformer.value:    T_val,
    voltage_source.value: V_val
}

A_mat_val = np.array(A.subs(subs_dict), dtype=np.float64)
B_mat_val = np.array(B.subs(subs_dict), dtype=np.float64)
C_mat_val = np.array(C.subs(subs_dict), dtype=np.float64)
D_mat_val = np.array(D.subs(subs_dict), dtype=np.float64)

x0_val = np.zeros_like(B_mat_val)

sys = ctrl.ss(A_mat_val, B_mat_val, C_mat_val, D_mat_val)
T, yout = ctrl.step_response(sys, T=25.0e-3, X0=x0_val)
yout = np.squeeze(yout) # yout is n_outputs x 1 x n_timesteps as the C matrix is has a shape of (n_outputs, n_states), so we need to squeeze the output to n_outputs x n_timesteps

print(f"Max time step: {np.max(np.diff(T))}")

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel("time (s)")
ax1.set_ylabel("Efforts (V, N, Pa)")
ax2.set_ylabel("Currents (A, m/s, m^3/s)")

for i, signal in enumerate(yout):
    if i < n_outputs // 2:
        ax1.plot(T, signal, label=f"e_{i}")
    else:
        ax2.plot(T, signal, label=f"f_{i - n_outputs // 2}")
    
ax1.legend()
ax2.legend()
plt.show()
