import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt

from .core import Node, StatefulElement, Bond, ElementOnePort, ElementTwoPort, Junction
from .elements import SourceEffort, SourceFlow, OneJunction, ZeroJunction

type SolutionType = dict[sp.Expr, sp.Expr]


class BondGraph:
    def __init__(self):
        self.elements = set()
        self.bonds: set[Bond] = set()

        self.state_vars: list[sp.Expr] = []
        self.equations: list[sp.Expr] = []
        self.inputs: list[sp.Symbol] = []

        self.solution: SolutionType = None

    def add_bond(self, bond: Bond) -> None:
        self.bonds.add(bond)

        for element in bond.elements:
            self.elements.add(element)

            if isinstance(element, SourceEffort) or isinstance(element, SourceFlow):
                self.inputs.append(element.value)

    def __handle_bonds(self) -> None:
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
                        # bond is strong bond for one junction
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

        for bond in self.bonds:
            handle_bond_element(bond.from_element, bond)
            handle_bond_element(bond.to_element, bond)

    def __handle_equations(self) -> None:
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

    def get_solution_equations(self) -> SolutionType:
        self.__handle_bonds()
        self.__handle_equations()
        Bond.counter = 0  # Reset bond counter for future bond graphs after solution has been computed --> ToDo: fix this hacky solution

        state_derivatives = [sp.Derivative(var, "t") for var in self.state_vars]
        self.solution = sp.solve(
            self.equations,
            state_derivatives + [b.effort for b in self.bonds] + [b.flow for b in self.bonds],
        )
        return self.solution

    def get_state_space(self) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, int, int, int]:
        # Retrieve solution if needed
        if self.solution is None:
            self.get_solution_equations()

        n_states = len(self.state_vars)
        n_inputs = len(self.inputs)  # Number of inputs (sources)
        n_outputs = 2 * len(self.bonds)  # effort & flow for each bond

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
            state_deriv = sp.Derivative(state_var, "t")  # symbolic derivative dx/dt
            f[i] = self.solution[state_deriv]

        h: sp.Matrix = sp.zeros(n_outputs, 1)  # efforts then flows
        for i, bond in enumerate(self.bonds):
            h[i] = self.solution[bond.effort]
            h[i + n_outputs // 2] = self.solution[bond.flow]

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

        for bond in self.bonds:
            G.add_edge(bond.from_element.name, bond.to_element.name, label=bond.num)

        # https://networkx.org/documentation/stable/auto_examples/graph/plot_dag_layout.html
        if nx.is_directed_acyclic_graph(G):
            for layer, nodes in enumerate(nx.topological_generations(G)):
                # `multipartite_layout` expects the layer as a node attribute, so add the
                # numeric layer value as a node attribute
                for node in nodes:
                    G.nodes[node]["layer"] = layer

            pos = nx.multipartite_layout(G, subset_key="layer")
        else:
            pos = layout(G, **kwargs)

        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=2000, 
            node_color="lightblue",
            font_size=10,
            font_color="black",
            arrows=True,
        )
        return fig, ax
