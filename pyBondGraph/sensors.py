from .core import StatefulElement, ElementOnePort

import sympy as sp


class IntegratedEffortSensor(ElementOnePort, StatefulElement):
    """Represents a sensor that integrates effort to its internal state variable."""

    def __init__(self, name: str):
        """Create a sensor that integrates effort to its internal state variable.

        Parameters
        ----------
        name : str
            The name of the element. Forwarded to the `Node` base class.
        """
        super().__init__(name, "")

    @property
    def state_var(self) -> sp.Symbol:
        """Returns the symbolic state variable (integrated effort) associated with the sensor.

        Returns
        -------
        sp.Symbol
            The symbolic state variable representing the integrated effort.
        """
        return sp.Symbol(f"e_int_{self.name}", real=True)

    @property
    def equations(self) -> list[sp.Expr]:
        """Returns the symbolic equations defining the behavior of the sensor that integrates effort.
        Formulated in the derivative of the internal state_var, stemming from integral causality.

        Returns
        -------
        list[sp.Expr]
            A list of symbolic equations representing the behavior of the sensor.
        """

        # Dynamic equation: e = d(state_var)/dt, f = 0
        return [
            self.bond.effort - sp.Derivative(self.state_var, "t"),
            self.bond.flow
        ]

class IntegratedFlowSensor(ElementOnePort, StatefulElement):
    """Represents a sensor that integrates flow to its internal state variable."""

    def __init__(self, name: str):
        """Create a sensor that integrates effort to its internal state variable.

        Parameters
        ----------
        name : str
            The name of the element. Forwarded to the `Node` base class.
        """
        super().__init__(name, "")

    @property
    def state_var(self) -> sp.Symbol:
        """Returns the symbolic state variable (integrated flow) associated with the sensor.

        Returns
        -------
        sp.Symbol
            The symbolic state variable representing the integrated flow.
        """
        return sp.Symbol(f"f_int_{self.name}", real=True)

    @property
    def equations(self) -> list[sp.Expr]:
        """Returns the symbolic equations defining the behavior of the sensor that integrates flow.
        Formulated in the derivative of the internal state_var, stemming from integral causality.

        Returns
        -------
        list[sp.Expr]
            A list of symbolic equations representing the behavior of the sensor.
        """

        # Dynamic equation: f = d(state_var)/dt, e = 0
        return [
            self.bond.flow - sp.Derivative(self.state_var, "t"),
            self.bond.effort
        ]