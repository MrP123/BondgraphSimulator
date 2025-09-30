from pyBondGraph import BondGraph, Bond, SourceEffort, Inductor, Capacitor, Resistor, OneJunction, ZeroJunction
from pyBondGraph.core import Junction

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

import itertools

bond_graph = BondGraph()

force_input = SourceEffort("F", "F")

mass_1 = Inductor("m1", "m1")
mass_2 = Inductor("m2", "m2")
spring_1 = Capacitor("k1", "c1")
spring_2 = Capacitor("k2", "c2")
damper_1 = Resistor("d1", "r1")
damper_2 = Resistor("d2", "r2")

junction_mass_1 = OneJunction("J1_1")
junction_mass_2 = OneJunction("J1_2")
junction_relative = ZeroJunction("J0_r")
junction_spring_damper = OneJunction("J1_r")

bond_graph.add_bond(Bond(force_input, junction_mass_2, "effort_out"))
bond_graph.add_bond(Bond(junction_mass_2, mass_2, "effort_out"))
bond_graph.add_bond(Bond(junction_mass_2, junction_relative, "flow_out"))
bond_graph.add_bond(Bond(junction_relative, junction_spring_damper, "flow_out"))
bond_graph.add_bond(Bond(junction_spring_damper, spring_2, "flow_out"))
bond_graph.add_bond(Bond(junction_spring_damper, damper_2, "flow_out"))
bond_graph.add_bond(Bond(junction_relative, junction_mass_1, "effort_out"))
bond_graph.add_bond(Bond(junction_mass_1, mass_1, "effort_out"))
bond_graph.add_bond(Bond(junction_mass_1, spring_1, "flow_out"))
bond_graph.add_bond(Bond(junction_mass_1, damper_1, "flow_out"))

A, B, C, D, n_states, n_inputs, n_outputs = bond_graph.get_state_space()

# Print results
print("x_vec:")
sp.pprint(bond_graph.state_vars)
print("u_vec:")
sp.pprint(bond_graph.inputs)
print("Matrix A:")
sp.pprint(A)
print("\nMatrix B:")
sp.pprint(B)
print("\nMatrix C:")
sp.pprint(C)
print("\nMatrix D:")
sp.pprint(D)


fig, ax = bond_graph.plot()
fig.show()


F_val = 1.0
m1_val = 2
m2_val = 1
k1_val = 250
k2_val = 400
c1_val = 5
c2_val = 15

# ToDo: maybe make numeric value part of the element class?
subs_dict = {
    force_input.value: F_val,
    mass_1.value: m1_val,
    mass_2.value: m2_val,
    spring_1.value: 1/k1_val,
    spring_2.value: 1/k2_val,
    damper_1.value: c1_val,
    damper_2.value: c2_val,
}

assert len(subs_dict) == len([elem for elem in bond_graph.elements if not isinstance(elem, Junction)]), "All elements (except junctions) must have numeric values for simulation!"

def to_numpy(M: sp.Matrix, subs: dict) -> np.ndarray:
    return np.array(M.subs(subs), dtype=np.float64)

A_mat_val = to_numpy(A, subs_dict)
B_mat_val = to_numpy(B, subs_dict)
C_mat_val = to_numpy(C, subs_dict)
D_mat_val = to_numpy(D, subs_dict)

lambda_real_abs = np.abs(np.real(np.linalg.eigvals(A_mat_val)))
stiffness_ratio = np.max(lambda_real_abs) / np.min(lambda_real_abs[np.nonzero(lambda_real_abs)])
print(f"Stiffness ratio: {stiffness_ratio:.2e}")

x0_val = np.zeros_like(A_mat_val[0, :])

t_sim = np.linspace(0, 5, 10000)
u_sim = np.zeros((n_inputs, t_sim.shape[0]))

# constant value inputs --> step
u_sim[0, 0:2500] = 0.0
u_sim[0, 2500:7500] = F_val
u_sim[0, 7500:] = 0.0

sys = ctrl.ss(A_mat_val, B_mat_val, C_mat_val, D_mat_val)
time_response: ctrl.TimeResponseData = ctrl.forced_response(sys, T=t_sim, U=u_sim, X0=x0_val)

T, yout, xout = time_response.time, time_response.outputs, time_response.states
yout = np.squeeze(yout)  # yout is n_outputs x 1 x n_timesteps as the C matrix is has a shape of (n_outputs, n_states), so we need to squeeze the output to n_outputs x n_timesteps

print(f"Max time step: {np.max(np.diff(T))}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)

ax1: plt.Axes
ax2: plt.Axes
ax3: plt.Axes

ax1.set_xlabel("time (s)")
ax1.set_ylabel("Efforts (V, N, Nm, Pa)")
ax2.set_ylabel("Currents (A, m/s, rad/s, m^3/s)")
ax3.set_ylabel("States")

markers_effort, markers_flow = itertools.tee(itertools.cycle(("+", "o", "*", "x", "s", "d")), 2) #cycle is not resettable --> 2 iterators
start_marker = itertools.cycle([11, 19, 23, 31, 41, 47, 53, 61, 71, 79, 83, 89, 97])

for i, signal in enumerate(yout):
    if i < n_outputs // 2:
        ax1.plot(T, signal, label=f"e_{i}", marker=next(markers_effort), markevery=(next(start_marker), 500))
    else:
        ax2.plot(T, signal, label=f"f_{i - n_outputs // 2}", marker=next(markers_flow), markevery=(next(start_marker), 500))

ax3.plot(T, xout.T, label=[sp.pretty(st) for st in bond_graph.state_vars])

ax1.legend()
ax2.legend()
ax3.legend()
plt.show()
