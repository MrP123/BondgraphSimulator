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
        self.value = sp.Symbol(value)
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
        self.value = sp.Symbol(value)
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

    def add_element(self, element: Node):
        self.elements.append(element)

    def add_connection(self, bond: Bond):
        self.connections.append(bond)

    def handle_bonds(self):
        for bond in self.connections:
            
            def handle_bond_element(element: Node, bond: Bond):
                if isinstance(element, ElementOnePort):
                    element.bond = bond #Assign bond to element
                
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
                                print(f"Assigned strong bond {bond} to OneJunction {element}.")
                            else:
                                raise ValueError(f"ZeroJunction {element} already has a strong bond: {element.strong_bond}. Cannot assign {bond}.")

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

    def get_state_equations(self):
        state_derivatives = [sp.Derivative(var, 't') for var in self.state_vars]

        solution = sp.solve(self.equations, state_derivatives + [b.effort for b in self.connections] + [b.flow for b in self.connections])
        return solution

bond_graph = BondGraph()

#https://en.wikipedia.org/wiki/Bond_graph#:~:text=denotes%20preferred%20causality.-,For%20the%20example%20provided,-%2C
voltage_source = SourceEffort("V", "V(t)")  # Spannungsquelle
capacitor = Capacitor("C6", "C6")             # Kapazität
inductor = Inductor("I3", "L3")               # Induktivität

resistor2 = Resistor("R2", "R2")               # Widerstand
resistor7 = Resistor("R7", "R7")               # Widerstand

transformer = Transformer("T1", "T1")           # Transformator

junction1 = OneJunction("J1")                 # 1-Junction
junction0 = ZeroJunction("J0")                 # 0-Junction

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

solution = bond_graph.get_state_equations()

# Determine A, B, C, D matrices for SS representation
n_states = len(bond_graph.state_vars)
n_inputs = 1
n_outputs = 1

A = sp.zeros(n_states, n_states)
B = sp.zeros(n_states, n_inputs)

# Extract the state derivatives (dx/dt) from the solution
for i, var in enumerate(bond_graph.state_vars):
    derivative = sp.Derivative(var, 't')  # symbolic derivatives

    if derivative in solution:  # check if they are in the solution
        expr: sp.Expr = solution[derivative]  # expression for dx/dt
        for j, state_var in enumerate(bond_graph.state_vars):
            A[i, j] = expr.diff(state_var)  # partial derivative w.r.t. state variables as A matrix links dx/dt with x via dx/dt ~ A*x

        input_counter = 0
        for element in bond_graph.elements:

            # Only check source elements for inputs
            if isinstance(element, SourceEffort):
                if element.bond.effort in solution:
                    input_var = solution[element.bond.effort]

                    B[i, input_counter] = expr.diff(input_var) # partial derivatives of dx/dt w.r.t. inputs as dx/dt ~ B*u
                    input_counter += 1
            
            #input_var = solution[efforts["V"]]
            #B[i, 0] = expr.diff(input_var)  

for element in bond_graph.elements:
    # Only check non-source elements
    if not isinstance(element, SourceEffort) and not isinstance(element, Junction):

        # Get output matrices C for effort and flow in element
        if element.bond.effort in solution:
            C_effort = sp.zeros(n_outputs, n_states)
            for j, state_var in enumerate(bond_graph.state_vars):
                C_effort[0, j] = solution[element.bond.effort].diff(state_var)
                
        if element.bond.flow in solution:
            C_flow = sp.zeros(n_outputs, n_states)
            for j, state_var in enumerate(bond_graph.state_vars):
                C_flow[0, j] = solution[element.bond.flow].diff(state_var)

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
