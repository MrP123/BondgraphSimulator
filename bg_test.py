from bondgraph.core import Bond, BondGraph
from bondgraph.junctions import JunctionEqualFlow, JunctionEqualEffort
from bondgraph.elements import Element_R, Element_I, Element_C, Source_effort

import sympy
from sympy import Symbol, integrate
import sympy.physics
from sympy.physics.mechanics import dynamicsymbols

voltage = Source_effort("voltage", dynamicsymbols("U"))
resistor = Element_R("resistor", Symbol("R"))
capacitor = Element_C("capacitor", Symbol("C"), integrate(dynamicsymbols("i"))) #was q --> q_dot is i
inductor = Element_I("inductor", Symbol("L"), integrate(dynamicsymbols("u")))   #was psi --> psi_dot is u
node = JunctionEqualFlow("node")

graph = BondGraph()
graph.add(Bond(voltage, node))
graph.add(Bond(node, resistor))
graph.add(Bond(node, capacitor))
graph.add(Bond(node, inductor))

state_equations = graph.get_state_equations()
print(f"State equations: {state_equations}")

#eqs = []
#states = []
#for s, rhs in state_equations.items():
#    s = s.diff()
#    eq = sympy.Equality(s, rhs)
#    eqs.append(eq)
#    states.append(s)

A, Bu = sympy.linear_eq_to_matrix(list(state_equations.values()), list(state_equations.keys()))
B, _ = sympy.linear_eq_to_matrix(-Bu, dynamicsymbols("U"))

print(f"A: {A}")
print(f"B: {B}")

#from bondgraph.visualization import gen_graphviz
#output = gen_graphviz(graph)
#output.view()