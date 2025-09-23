from pyBG import BondGraph, Bond, SourceEffort, Inductor, Capacitor, Resistor, Transformer, OneJunction, ZeroJunction

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

bond_graph = BondGraph()

# https://en.wikipedia.org/wiki/Bond_graph#:~:text=denotes%20preferred%20causality.-,For%20the%20example%20provided,-%2C
voltage_source = SourceEffort("V", "V(t)")
capacitor = Capacitor("C6", "C6")
inductor = Inductor("I3", "L3")
resistor2 = Resistor("R2", "R2")
resistor7 = Resistor("R7", "R7")
transformer = Transformer("T1", "T1")
junction1 = OneJunction("J1")
junction0 = ZeroJunction("J0")

bond_graph.add_bond(Bond(voltage_source, junction1, "effort_out"))
bond_graph.add_bond(Bond(junction1, resistor2, "flow_out"))
bond_graph.add_bond(Bond(junction1, inductor, "effort_out"))
bond_graph.add_bond(Bond(junction1, transformer, "flow_out"))
bond_graph.add_bond(Bond(transformer, junction0, "flow_out"))
bond_graph.add_bond(Bond(junction0, capacitor, "flow_out"))
bond_graph.add_bond(Bond(junction0, resistor7, "effort_out"))

A, B, C, D, n_states, n_inputs, n_outputs = bond_graph.get_state_space()

# Print results
print("Matrix A:")
sp.pprint(A)
print("\nMatrix B:")
sp.pprint(B)
print("\nMatrix C:")
sp.pprint(C)
print("\nMatrix D:")
sp.pprint(D)

R_val = 470.0
C_val = 10e-6
L_val = 1e-3
V_val = 1.0
T_val = 2.0

# ToDo: maybe make numeric value part of the element class?
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

# Initial conditions for the state variables --> not yet working
#x0_entries = sp.symbols(f"x0_0:{n_states}", real=True)
#y0_entries = sp.symbols(f"y0_0:{n_outputs}", real=True)
#x0 = sp.Matrix(x0_entries)
#y0 = sp.Matrix(y0_entries)
#U0_C6 = sp.symbols('U0_C6', real=True)
#U0_I3 = sp.symbols('U0_I3', real=True)

#for i, bond in enumerate(bond_graph.connections):
#    if bond.effort == capacitor.bond.effort:
#        print(f"capacitor = {i}")
#    if bond.flow == inductor.bond.flow:
#        print(f"inductor = {i}")

#sp.solve(C[[2, 5], :] * x0 - sp.Matrix([U0_I3, U0_C6]), x0)

fig, ax = bond_graph.plot()
fig.show()

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
