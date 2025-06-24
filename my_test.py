import sympy as sp

# Beispiel: Bondgraph-Datenstruktur
bond_graph = {
    "elements": [
        {"type": "Se", "name": "V", "value": "V(t)"},  # Spannungsquelle
        {"type": "C", "name": "C", "value": "C"},     # Kapazität
        {"type": "I", "name": "L", "value": "L"},     # Induktivität
        {"type": "R", "name": "R", "value": "R"}      # Widerstand
    ],
    "connections": [
        ("V", "1_1"),  # Spannungsquelle an 1-Junction
        ("1_1", "C"),  # 1-Junction an Kondensator
        ("1_1", "R"),  # 1-Junction an Widerstand
        ("1_1", "L"),  # 1-Junction an Induktivität
    ]
}

import sympy as sp

# Funktion zur Ableitung der Zustandsraumdarstellung
def derive_state_space(bond_graph):
    # Symbolische Variablen für Zustandsgrößen, Flüsse und Anstrengungen
    state_vars = {}  # Zustandsgrößen (z. B. q_C, p_L)
    efforts = {}     # Anstrengungen (e_C, e_L, ...)
    flows = {}       # Flüsse (f_C, f_L, ...)
    equations = []   # Liste der Gleichungen

    # Zustandsgrößen für C- und I-Elemente
    for element in bond_graph["elements"]:
        name = element["name"]

        if element["type"] == "C":
            state_vars[name] = sp.Symbol(f"q_{name}")  # Ladung
            efforts[name] = sp.Symbol(f"e_{name}")    # Anstrengung
            flows[name] = sp.Symbol(f"f_{name}")      # Fluss

            # Dynamische Gleichung: f = dq/dt, e = q / C
            equations.append(flows[name] - sp.Derivative(state_vars[name], 't'))
            equations.append(efforts[name] - state_vars[name] / sp.Symbol(element["value"]))

        elif element["type"] == "I":
            state_vars[name] = sp.Symbol(f"p_{name}")  # Impuls
            efforts[name] = sp.Symbol(f"e_{name}")    # Anstrengung
            flows[name] = sp.Symbol(f"f_{name}")      # Fluss
            # Dynamische Gleichung: e = dp/dt, f = p / L
            equations.append(efforts[name] - sp.Derivative(state_vars[name], 't'))
            equations.append(flows[name] - state_vars[name] / sp.Symbol(element["value"]))

        elif element["type"] == "R":
            efforts[name] = sp.Symbol(f"e_{name}")    # Anstrengung
            flows[name] = sp.Symbol(f"f_{name}")      # Fluss
            # Widerstand: e = R * f
            equations.append(efforts[name] - sp.Symbol(element["value"]) * flows[name])

        elif element["type"] == "Se":
            efforts[name] = sp.Symbol(f"e_{name}")    # Anstrengung
            flows[name] = sp.Symbol(f"f_{name}")      # Fluss
            # Quelle: e = konstante Anstrengung
            equations.append(efforts[name] - sp.Symbol(element["value"]))

    # 0- und 1-Junctions
    junctions = {}
    for connection in bond_graph["connections"]:
        from_element, to_element = connection

        if "0_" in from_element or "0_" in to_element:  # 0-Junction
            junction_name = from_element if "0_" in from_element else to_element
            
            if junction_name not in junctions:
                junctions[junction_name] = {"efforts": [], "flows": []}
            if "0_" in from_element:
                junctions[junction_name]["flows"].append(flows[to_element])
                junctions[junction_name]["efforts"].append(efforts[to_element])
            else:
                junctions[junction_name]["flows"].append(flows[from_element])
                junctions[junction_name]["efforts"].append(efforts[from_element])

        elif "1_" in from_element or "1_" in to_element:  # 1-Junction
            junction_name = from_element if "1_" in from_element else to_element
            if junction_name not in junctions:
                junctions[junction_name] = {"efforts": [], "flows": []}
            if "1_" in from_element:
                junctions[junction_name]["flows"].append(flows[to_element])
                junctions[junction_name]["efforts"].append(efforts[to_element])
            else:
                junctions[junction_name]["flows"].append(flows[from_element])
                junctions[junction_name]["efforts"].append(efforts[from_element])

    # Gleichungen für Junctions
    for junction_name, junction_data in junctions.items():
        if "0_" in junction_name:  # 0-Junction: Gemeinsame Anstrengung, Flüsse summieren sich
            equations.append(sum(junction_data["flows"]))
            for e in junction_data["efforts"][1:]:
                equations.append(junction_data["efforts"][0] - e)
        elif "1_" in junction_name:  # 1-Junction: Gemeinsamer Fluss, Anstrengungen summieren sich
            equations.append(sum(junction_data["efforts"]))
            for f in junction_data["flows"][1:]:
                equations.append(junction_data["flows"][0] - f)

    # Symbolische Lösung der Gleichungen
    state_vars_list = list(state_vars.values())
    derivatives = [sp.Derivative(var, 't') for var in state_vars_list]
    solution = sp.solve(equations, derivatives + list(efforts.values()) + list(flows.values()))

    print(solution)

    # Matrizen A, B, C, D ableiten
    A = sp.zeros(len(state_vars_list), len(state_vars_list))
    B = sp.zeros(len(state_vars_list), 1)

    # Ableitungen der Zustandsgrößen (dx/dt) aus der Lösung extrahieren
    for i, var in enumerate(state_vars_list):
        derivative = sp.Derivative(var, 't')  # Symbolische Ableitung
        if derivative in solution:  # Prüfen, ob die Ableitung in der Lösung enthalten ist
            expr = solution[derivative]  # Ausdruck für dx/dt
            for j, state_var in enumerate(state_vars_list):
                A[i, j] = expr.diff(state_var)  # Partielle Ableitung nach Zustandsvariablen

            input_counter = 0
            for element in bond_graph["elements"]:
                if element["type"] == "Se": # Effort source

                    effort = efforts[element["name"]]
                    if effort in solution:
                        input_var = solution[effort]

                        B[i, input_counter] = expr.diff(input_var) # Partielle Ableitung nach Eingang (z. B. V)
                        input_counter += 1

            #input_var = solution[efforts["V"]]
            #B[i, 0] = expr.diff(input_var)  

    for element in bond_graph["elements"]:

        if element["type"] == "C":
            name = element["name"]
            elem_effort = sp.Symbol(f"e_{name}")
            elem_flow = sp.Symbol(f"f_{name}")

            if elem_effort in solution:
                C_voltage_C = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_voltage_C[0, j] = solution[elem_effort].diff(state_var)
                
            if elem_flow in solution:
                C_current_C = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_current_C[0, j] = solution[elem_flow].diff(state_var)

        elif element["type"] == "I":
            name = element["name"]
            elem_effort = sp.Symbol(f"e_{name}")
            elem_flow = sp.Symbol(f"f_{name}")

            if elem_effort in solution:
                C_voltage_I = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_voltage_I[0, j] = solution[elem_effort].diff(state_var)
                
            if elem_flow in solution:
                C_current_I = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_current_I[0, j] = solution[elem_flow].diff(state_var)

        elif element["type"] == "R":
            name = element["name"]
            elem_effort = sp.Symbol(f"e_{name}")
            elem_flow = sp.Symbol(f"f_{name}")

            if elem_effort in solution:
                C_voltage_R = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_voltage_R[0, j] = solution[elem_effort].diff(state_var)
                
            if elem_flow in solution:
                C_current_R = sp.zeros(1, len(state_vars_list))
                for j, state_var in enumerate(state_vars_list):
                    C_current_R[0, j] = solution[elem_flow].diff(state_var)

    # Matrix C und D
    C_mats = {
        "voltage_C": C_voltage_C,
        "current_C": C_current_C,
        "voltage_I": C_voltage_I,
        "current_I": C_current_I,
        "voltage_R": C_voltage_R,
        "current_R": C_current_R,
    }


    D = sp.zeros(1, 1)

    return A, B, C_mats, D

# Zustandsraumdarstellung ableiten
A, B, C_mats, D = derive_state_space(bond_graph)

# Ergebnisse anzeigen
print("Matrix A:")
sp.pprint(A)
print("\nMatrix B:")
sp.pprint(B)
print("\nMatrix C:")
for key, C_l in C_mats.items():
    print(f"\n{key}:")
    sp.pprint(C_l)
print("\nMatrix D:")
sp.pprint(D)


x0 = sp.MatrixSymbol("x0", A.shape[0], 1)  # Anfangszustand als Matrix

init_cond = sp.Eq(sp.Symbol("Uc"), (C_mats["voltage_C"] * x0)[0])
print("\nInitial condition:")
sp.pprint(init_cond)

init_cond_solution = sp.solve(init_cond, x0[0, 0])
print("\nSolution for initial condition:")
sp.pprint(init_cond_solution)


import numpy as np

R_val = 470
C_val = 10e-6
L_val = 1e-3
V_val = 5
Uc_0_val = 5

subs_dict = {
    "R": R_val,
    "C": C_val,
    "L": L_val,
    "V(t)": V_val
}

A_mat_val = np.array(A.subs(subs_dict), dtype=np.float64)
B_mat_val = np.array(B.subs(subs_dict), dtype=np.float64)
C_mat_val = {key: np.array(val.subs(subs_dict), dtype=np.float64) for key, val in C_mats.items()}
D_mat_val = np.array(D.subs(subs_dict), dtype=np.float64)

x0_val = np.zeros_like(B_mat_val)
#x0_val[0, 0] = init_cond_solution[0].subs({"Uc": Uc_0_val, "C": C_val})
#x0_val = np.array(x0_val, dtype=np.float64)

import control as ctrl
import matplotlib.pyplot as plt

print(f"Running simulation with\n A:\n{A_mat_val}\nB:\n{B_mat_val}\nC:\n{C_mat_val}\nD:\n{D_mat_val}\nx0:\n{x0_val}")

sys = ctrl.ss(A_mat_val, -B_mat_val, C_mat_val["voltage_C"], -D_mat_val)
T, yout = ctrl.step_response(sys, T=50.0e-3, X0=x0_val)

plt.plot(T, yout)
plt.title("Step Response of the Bond Graph System")
plt.show()