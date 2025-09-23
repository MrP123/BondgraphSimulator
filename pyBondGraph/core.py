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

        self.effort = sp.Symbol(f"e_{self.num}", real=True)
        self.flow = sp.Symbol(f"f_{self.num}", real=True)

    @property
    def elements(self):
        return (self.from_element, self.to_element)

    def __repr__(self):
        return f"Bond(from={self.from_element}, to={self.to_element}, causality={self.causality})"


class Junction(Node, ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self.bonds: list[Bond] = []
        self.strong_bond = None

    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        raise NotImplementedError("Subclasses should implement this method")


class ElementOnePort(Node, ABC):
    def __init__(self, name: str, value: str):
        super().__init__(name)
        self.value = sp.Symbol(value, real=True, positive=True)  # Ensure value is a positive real number
        self.bond: Bond = None  # bond that connects this element to a bond graph

    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        raise NotImplementedError("Subclasses should implement this method")

    def __repr__(self):
        return f"{self.__class__.__name__}(name = {self.name}, value = {self.value})"


class ElementTwoPort(Node, ABC):
    def __init__(self, name: str, value: str):
        super().__init__(name)
        self.value = sp.Symbol(value, real=True, positive=True)
        self.bond1: Bond = None  # ElementOther   --(bond1)--> ElementTwoPort
        self.bond2: Bond = None  # ElementTwoPort --(bond2)--> ElementOther

    @property
    @abstractmethod
    def equations(self) -> list[sp.Expr]:
        raise NotImplementedError("Subclasses should implement this method")

    def __repr__(self):
        return f"{self.__class__.__name__}(name = {self.name}, value = {self.value})"
