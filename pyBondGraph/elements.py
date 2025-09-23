from .core import StatefulElement, ElementOnePort, ElementTwoPort, Junction

import sympy as sp


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
        return sp.Symbol(f"q_{self.name}", real=True)

    @property
    def equations(self) -> list[sp.Expr]:
        # Dynamic equation: f = dq/dt, e = q / C
        return [
            self.bond.flow - sp.Derivative(self.state_var, "t"),
            self.bond.effort - self.state_var / self.value,
        ]


class Inductor(ElementOnePort, StatefulElement):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def state_var(self) -> sp.Symbol:
        return sp.Symbol(f"p_{self.name}", real=True)

    @property
    def equations(self) -> list[sp.Expr]:
        # Dynamic equation: e = dp/dt, f = p / L
        return [
            self.bond.effort - sp.Derivative(self.state_var, "t"),
            self.bond.flow - self.state_var / self.value,
        ]


class Resistor(ElementOnePort):
    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        # Resistance: e = R * f
        if self.bond.causality == "effort_out":
            return [self.bond.flow - 1 / self.value * self.bond.effort]
        elif self.bond.causality == "flow_out":
            return [self.bond.effort - self.value * self.bond.flow]


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
            return [
                self.bond1.flow - 1 / self.value * self.bond2.flow,
                self.bond2.effort - 1 / self.value * self.bond1.effort,
            ]
        elif self.bond1.causality == "flow_out":
            return [
                self.bond1.effort - self.value * self.bond2.effort,
                self.bond2.flow - self.value * self.bond1.flow,
            ]


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
            return [
                self.bond1.flow - 1 / self.value * self.bond2.effort,
                self.bond2.flow - 1 / self.value * self.bond1.effort,
            ]
        elif self.bond1.causality == "flow_out":
            return [
                self.bond1.effort - self.value * self.bond2.flow,
                self.bond2.effort - self.value * self.bond1.flow,
            ]


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
