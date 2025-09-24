from pyBondGraph import BondGraph, Bond, SourceEffort, Inductor, Resistor, OneJunction, Gyrator

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

bond_graph = BondGraph()

voltage_source = SourceEffort("V", "U_A(t)")
junction_elec = OneJunction("J0_1")
inductor = Inductor("I_elec", "L_A")
resistor = Resistor("R_elec", "R_A")
gyrator = Gyrator("G1", "K_t")
junction_mech = OneJunction("J1_1")
bearing = Resistor("R_mech", "R_B")
inertia = Inductor("I_mech", "J")

bond_graph.add_bond(Bond(voltage_source, junction_elec, "effort_out"))
bond_graph.add_bond(Bond(junction_elec, resistor, "flow_out"))
bond_graph.add_bond(Bond(junction_elec, inductor, "effort_out"))
bond_graph.add_bond(Bond(junction_elec, gyrator, "flow_out"))
bond_graph.add_bond(Bond(gyrator, junction_mech, "effort_out"))
bond_graph.add_bond(Bond(junction_mech, bearing, "flow_out"))
bond_graph.add_bond(Bond(junction_mech, inertia, "effort_out"))

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


fig, ax = bond_graph.plot()
fig.show()


U_A_val = 5.0
R_A_val = 4
L_A_val = 15e-6
K_t_val = 9.54e-3
J_val = 1e-6
R_B_val = 1e-6

# ToDo: maybe make numeric value part of the element class?
subs_dict = {
    voltage_source.value: U_A_val,
    resistor.value: R_A_val,
    inductor.value: L_A_val,
    gyrator.value: K_t_val,
    inertia.value: J_val,
    bearing.value: R_B_val,
}


def to_numpy(M: sp.Matrix, subs: dict) -> np.ndarray:
    return np.array(M.subs(subs), dtype=np.float64)


A_mat_val = to_numpy(A, subs_dict)
B_mat_val = to_numpy(B, subs_dict)
C_mat_val = to_numpy(C, subs_dict)
D_mat_val = to_numpy(D, subs_dict)

x0_val = np.zeros_like(A_mat_val[0, :])

sys = ctrl.ss(A_mat_val, B_mat_val, C_mat_val, D_mat_val)
T, yout = ctrl.step_response(sys, T=0.5, X0=x0_val)
yout = np.squeeze(yout)  # yout is n_outputs x 1 x n_timesteps as the C matrix is has a shape of (n_outputs, n_states), so we need to squeeze the output to n_outputs x n_timesteps

print(f"Max time step: {np.max(np.diff(T))}")

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlabel("time (s)")
ax1.set_ylabel("Efforts (V, N, Nm, Pa)")
ax2.set_ylabel("Currents (A, m/s, rad/s, m^3/s)")

for i, signal in enumerate(yout):
    if i < n_outputs // 2:
        ax1.plot(T, signal, label=f"e_{i}")
    else:
        ax2.plot(T, signal, label=f"f_{i - n_outputs // 2}")

ax1.legend()
ax2.legend()
plt.show()
