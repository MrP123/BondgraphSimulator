from abc import ABC, abstractmethod

import sympy as sp


class StatefulElement(ABC):
    """Base class for all stateful elements (capacitor, inductor) in the bond graph.
    Requires implementation of a `state_var` property that returns the symbolic state variable associated with the element.
    """
    @property
    @abstractmethod
    def state_var(self) -> sp.Symbol:
        """sp.Symbol: Returns the symbolic state variable associated with the stateful element."""
        raise NotImplementedError("Subclasses should implement this method")


class Node(ABC):
    """Base class for all nodes in the bond graph."""

    def __init__(self, name: str):
        """Base class for all nodes in the bond graph.

        Parameters
        ----------
        name : str
            The name of the node. Will be shown on the bond graph plot.
        """
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class Bond:
    """Represents a bond between two elements in the bond graph."""

    counter = 0

    def __init__(self, from_element: Node, to_element: Node, causality: str):
        """Create a bond between two elements with specified causality.
        The positive direction of this power bond is from `from_element` to `to_element`.
        Efforts and flows are represented by symbolic `sympy.Symbol`s that are strictly real-valued.

        Parameters
        ----------
        from_element : Node
            The element where the bond originates.
        to_element : Node
            The element where the bond terminates.
        causality : str
            The causality of the bond, either `effort_out` or `flow_out`.
            This definition is always from the perspective of the `from_element`.
            Therefore a `Bond(OneJunction(...), Inductor(...), "effort_out")` means that the
            `OneJunction` imposes effort on the `Inductor`, meaning it has an equivalent `effort_in` causality.
            Likewise a `Bond(OneJunction(...), Capacitor(...), "flow_out")` means that the `OneJunction` imposes
            flow on the `Capacitor`, meaning it has an equivalent `flow_in`/`effort_out` causality.

        Raises
        ------
        ValueError
            If the causality is not 'effort_out' or 'flow_out'.
        """

        self.from_element = from_element
        self.to_element = to_element
        self.causality = causality

        if causality not in ["effort_out", "flow_out"]:
            raise ValueError(f"Invalid causality: {causality}. Must be 'effort_out' or 'flow_out'.")

        self.num = Bond.counter
        Bond.counter += 1

        self.effort = sp.Symbol(f"e_{self.num}", real=True)
        self.flow = sp.Symbol(f"f_{self.num}", real=True)

    @property
    def elements(self) -> tuple[Node, Node]:
        """tuple[Node, Node]: The two elements connected by the bond. First element is `from_element`, second is `to_element`."""
        return (self.from_element, self.to_element)

    def __repr__(self) -> str:
        return f"Bond(from={self.from_element}, to={self.to_element}, causality={self.causality})"


class Junction(Node, ABC):
    """Base class for all junctions in the bond graph."""

    def __init__(self, name: str):
        """Base class for all junctions in the bond graph.
        Stores a list of associated bonds and a reference to the "strong" bond.

        Parameters
        ----------
        name : str
            The name of the junction. Will be shown on the bond graph plot.
        """
        super().__init__(name)
        self.bonds: list[Bond] = []
        self.strong_bond = None

    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        """list[sp.Expr]: Returns the symbolic equations defining the behavior of the junction."""
        raise NotImplementedError("Subclasses should implement this method")


class ElementOnePort(Node, ABC):
    """Base class for all one-port elements in the bond graph.
    Requires implementation of an `equations` property that returns the list of symbolic equations defining the element's behavior.
    This can also be just one equation.
    """

    def __init__(self, name: str, value: str):
        """ABC for a one-port element. Acts as base class for all one-port elements like Inductor, Capacitor, Resistor, SourceEffort, SourceFlow.
        Stores an associated bond and a symbolic value (real and positive) for its defining characteristics e.g. resistance, capacitance, etc.

        Parameters
        ----------
        name : str
            The name of the element. Will be shown on the bond graph plot.
        value : str
            The name of the value associated with the element, is internally used for creating a `sympy.Symbol`.
        """

        super().__init__(name)
        self.value = sp.Symbol(value, real=True, positive=True)  # Ensure value is a positive real number
        self.bond: Bond = None  # bond that connects this element to a bond graph

    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        """list[sp.Expr]: Returns the symbolic equations defining the behavior of the one-port element."""
        raise NotImplementedError("Subclasses should implement this method")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name = {self.name}, value = {self.value})"


class ElementTwoPort(Node, ABC):
    """Base class for all two-port elements in the bond graph.
    Requires implementation of an `equations` property that returns the list of symbolic equations defining the element's behavior.
    """

    def __init__(self, name: str, value: str):
        """ABC for a two-port element. Acts as base class for all two-port elements like Transformer, Gyrator.
        Stores two associated bonds and a symbolic value (real and positive) for its defining characteristics e.g. conversion factor, ratio, etc.

        Parameters
        ----------
        name : str
            The name of the element. Will be shown on the bond graph plot.
        value : str
            The name of the value associated with the element, is internally used for creating a `sympy.Symbol`.
        """

        super().__init__(name)
        self.value = sp.Symbol(value, real=True, positive=True)
        self.bond1: Bond = None  # ElementOther   --(bond1)--> ElementTwoPort
        self.bond2: Bond = None  # ElementTwoPort --(bond2)--> ElementOther

    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        """list[sp.Expr]: Returns the symbolic equations defining the behavior of the two-port element."""
        raise NotImplementedError("Subclasses should implement this method")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name = {self.name}, value = {self.value})"
