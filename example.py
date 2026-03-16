from two_phase import solve_two_phase
import numpy as np

# Problem data from the Markdown/Documentation
# Maximize 2x1 + 3x2 + 4x3
c = np.array([2.0, 3.0, 4.0])  
# Subject to:
# -x1 - 2x2 - x3 >= -10  => (multiplied by -1 and sense flipped in solver)
# 2x1 + x2 + 3x3 = 15
# x1 + x2 + x3 >= 6
A = np.array([[-1.0, -2.0, -1.0], [2.0, 1.0, 3.0], [1.0, 1.0, 1.0]])
b = np.array([-10.0, 15.0, 6.0])
senses = [">=", "=", ">="]

# Custom variable labeling
labels = ["Product_A", "Product_B", "Product_C"]

print("--- Running Two-Phase Simplex Solver ---")
sol = solve_two_phase(c, A, b, senses, objective="max", variable_labels=labels)

print(f"Status: {sol['status']}")
if sol['status'] == "optimal":
    print(f"Optimal Objective Value: {sol['objective']}")
    print(f"Solution Vector (Original Order): {sol['x']}")
    
    print("\nBasic Variables (BV):")
    for var, val in sol["bv"].items():
        print(f"  {var}: {val}")
        
    print("\nNon-Basic Variables (NBV):")
    for var, val in sol["nbv"].items():
        print(f"  {var}: {val}")

    print("\nCheck vs Markdown Result: Objective should be 25.0")
    if abs(sol['objective'] - 25.0) < 1e-9:
        print("MATCH: Result is consistent with the documentation.")
    else:
        print("MISMATCH: Please check problem definitions.")
