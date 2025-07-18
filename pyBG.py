from abc import ABC, abstractmethod

import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt

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
        self.elements = set()
        self.connections: set[Bond] = set()

        self.state_vars: list[sp.Expr] = []
        self.equations: list[sp.Expr] = []
        self.inputs: list[sp.Symbol] = []

    def add_bond(self, bond: Bond):
        self.connections.add(bond)
        
        for element in bond.elements:
            self.elements.add(element)

            if isinstance(element, SourceEffort) or isinstance(element, SourceFlow):
                self.inputs.append(element.value)

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
                self.equations.extend(element.equations)

                if isinstance(element, StatefulElement):
                    self.state_vars.append(element.state_var)

            elif isinstance(element, ElementTwoPort):
                bond1 = element.bond1
                bond2 = element.bond2
                if bond1 is None or bond2 is None:
                    raise ValueError(f"Element {element} has no connected bonds.")
                
                # Add equations from the element to the bond graph
                self.equations.extend(element.equations)

            elif isinstance(element, Junction):
                self.equations.extend(element.equations)

    def get_solution_equations(self):
        state_derivatives = [sp.Derivative(var, 't') for var in self.state_vars]

        self.solution = sp.solve(self.equations, state_derivatives + [b.effort for b in self.connections] + [b.flow for b in self.connections])
        return self.solution
    
    def get_state_space(self):
        if self.solution is None:
            raise ValueError("No solution available. Please call get_solution_equations() first.")

        n_states = len(self.state_vars)
        n_inputs = len(self.inputs)  # Number of inputs (sources)
        n_outputs = 2*len(self.connections) #effort & flow for each bond

        # General form of a state space model
        # x_dot = f(x, u)
        #     y = h(x, u)
        # Simplification for linear systems:
        # x_dot = A*x + B*u
        #     y = C*x + D*u
        # --> therefore
        # A = ∂f/∂x, B = ∂f/∂u, C = ∂h/∂x, D = ∂h/∂u each at stationary point 0

        f: sp.Matrix = sp.zeros(n_states, 1)
        for i, state_var in enumerate(self.state_vars):
            state_deriv = sp.Derivative(state_var, 't')  # symbolic derivative dx/dt
            f[i] = self.solution[state_deriv]

        h: sp.Matrix = sp.zeros(n_outputs, 1) # efforts then flows
        for i, bond in enumerate(self.connections):
            h[i] = self.solution[bond.effort]
            h[i + n_outputs//2] = self.solution[bond.flow]

        A = f.jacobian(self.state_vars)
        B = f.jacobian(self.inputs)

        C = h.jacobian(self.state_vars)
        D = h.jacobian(self.inputs)
        # alternatively could use sp.linear_eq_to_matrix(...)

        return A, B, C, D, n_states, n_inputs, n_outputs
    
    def plot(self, layout: callable = nx.spectral_layout, **kwargs):

        G = nx.DiGraph()

        for elem in self.elements:
            G.add_node(elem.name, label=elem.name)

        for bond in self.connections:
            G.add_edge(bond.from_element.name, bond.to_element.name, label=bond.num)

        pos = layout(G, **kwargs)

        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_color='black', arrows=True)
        return fig, ax