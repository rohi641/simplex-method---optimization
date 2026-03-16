from typing import List, Tuple
import numpy as np


def solve_two_phase(c: np.ndarray, A: np.ndarray, b: np.ndarray, senses: List[str], objective: str = "max",
                    variable_labels: List[str] = None, tol: float = 1e-9, maxiter: int = 1000):
    c = np.asarray(c, dtype=float).flatten()
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).flatten()
    m, n = A.shape

    if objective not in ("min", "max"):
        raise ValueError("objective must be 'min' or 'max'")

    if variable_labels is not None and len(variable_labels) != n:
        raise ValueError(f"variable_labels must have length {n}")

    # Keep original objective coefficients and convert maximize to minimization by negating c
    c_orig = c.copy()
    if objective == "max":
        c = -c

    # STEP 1: ensure b >= 0
    A2 = A.copy()
    b2 = b.copy()
    senses2 = senses.copy()
    for i in range(m):
        if b2[i] < -tol:
            A2[i, :] *= -1
            b2[i] *= -1
            if senses2[i] == "<=":
                senses2[i] = ">="
            elif senses2[i] == ">=":
                senses2[i] = "<="

    # Build standard-form matrix with slack/surplus/artificial
    if variable_labels:
        var_names = list(variable_labels)
    else:
        var_names = [f"x{j+1}" for j in range(n)]
    
    basis = []  # basis column indices
    artificial_indices = []

    A_aug = A2.copy()
    slack_count = 0
    surplus_count = 0
    art_count = 0

    # We'll build full augmented A matrix progressively
    for i in range(m):
        sense = senses2[i]
        if sense == "<=":
            # add slack variable
            col = np.zeros((m, 1))
            col[i, 0] = 1.0
            A_aug = np.hstack((A_aug, col))
            slack_count += 1
            var_names.append(f"s{slack_count}")
            basis.append(A_aug.shape[1] - 1)
        elif sense == ">=":
            # add surplus var (subtract) and artificial var
            col_s = np.zeros((m, 1))
            col_s[i, 0] = -1.0
            A_aug = np.hstack((A_aug, col_s))
            surplus_count += 1
            var_names.append(f"e{surplus_count}")

            # artificial
            col_a = np.zeros((m, 1))
            col_a[i, 0] = 1.0
            A_aug = np.hstack((A_aug, col_a))
            art_count += 1
            var_names.append(f"a{art_count}")
            artificial_indices.append(A_aug.shape[1] - 1)
            basis.append(A_aug.shape[1] - 1)
        elif sense == "=":
            # add artificial
            col_a = np.zeros((m, 1))
            col_a[i, 0] = 1.0
            A_aug = np.hstack((A_aug, col_a))
            art_count += 1
            var_names.append(f"a{art_count}")
            artificial_indices.append(A_aug.shape[1] - 1)
            basis.append(A_aug.shape[1] - 1)
        else:
            raise ValueError("senses must be one of '<=', '>=', '='")

    # Convert to numpy arrays and track counts
    A_aug = np.asarray(A_aug, dtype=float)
    total_vars = A_aug.shape[1]

    # Build Phase I tableau
    # Tableau rows: row0 (phase objective), then m constraint rows
    tableau = np.zeros((m + 1, total_vars + 1), dtype=float)
    tableau[1:, 0:total_vars] = A_aug
    tableau[1:, -1] = b2

    # Phase I objective: w' = sum(a_i). We set row0 coefficients to -1 for artificials
    tableau[0, :] = 0.0
    for ai in artificial_indices:
        tableau[0, ai] = -1.0
    tableau[0, -1] = 0.0

    # For every artificial in the basis, add its row to row0 to eliminate the artificial column
    for i_row in range(m):
        if basis[i_row] in artificial_indices:
            tableau[0, :] += tableau[i_row + 1, :]

    def pivot(t: np.ndarray, row: int, col: int):
        # row and col are tableau indices: row in [1..m], col in [0..total_vars-1]
        piv = t[row, col]
        if abs(piv) < tol:
            raise ZeroDivisionError("Pivot too small")
        t[row, :] = t[row, :] / piv
        for r in range(t.shape[0]):
            if r != row:
                t[r, :] -= t[r, col] * t[row, :]

    def choose_entering(t: np.ndarray):
        # For minimization: entering if tableau[0, j] > tol
        row0 = t[0, :-1]
        candidates = np.where(row0 > tol)[0]
        if candidates.size == 0:
            return None
        # Bland's rule could be used; choose largest positive to speed
        return int(candidates[np.argmax(row0[candidates])])

    def choose_leaving(t: np.ndarray, col: int):
        col_vals = t[1:, col]
        rhs = t[1:, -1]
        ratios = []
        for i in range(len(col_vals)):
            aij = col_vals[i]
            if aij > tol:
                ratios.append(rhs[i] / aij)
            else:
                ratios.append(np.inf)
        ratios = np.array(ratios)
        if np.all(np.isinf(ratios)):
            return None
        leaving = int(np.argmin(ratios))
        return leaving + 1  # tableau row index

    # Simplex loop for Phase I (minimize w')
    iter_count = 0
    status = ""
    while True:
        if iter_count > maxiter:
            status = "iteration_limit"
            break
        entering = choose_entering(tableau)
        if entering is None:
            status = "optimal_phase1"
            break
        leaving = choose_leaving(tableau, entering)
        if leaving is None:
            status = "unbounded"
            break
        # update basis
        basis[leaving - 1] = entering
        pivot(tableau, leaving, entering)
        iter_count += 1

    w_val = tableau[0, -1]
    if status == "unbounded":
        return {"status": "unbounded", "x": None, "objective": None, "bv": None, "nbv": None}

    if w_val > tol:
        return {"status": "infeasible", "x": None, "objective": None, "bv": None, "nbv": None}

    # Phase I success: remove artificial columns
    keep_cols = [j for j in range(total_vars) if j not in artificial_indices]
    phase2_var_names = [var_names[j] for j in keep_cols]
    
    # Map old indices to new
    col_map = {old: new for new, old in enumerate(keep_cols)}
    A_phase2 = tableau[:, keep_cols + [-1]]  # include RHS
    tableau = A_phase2.copy()
    total_vars = tableau.shape[1] - 1

    # Update basis indices to new column indices or mark if basic var was artificial
    new_basis = []
    for bi in basis:
        if bi in artificial_indices:
            # try to find a nonzero column in that row to pivot in
            row_idx = len(new_basis) + 1
            found = False
            for j in range(total_vars):
                if abs(tableau[row_idx, j]) > tol:
                    # pivot to replace artificial
                    try:
                        pivot(tableau, row_idx, j)
                    except ZeroDivisionError:
                        continue
                    new_basis.append(j)
                    found = True
                    break
            if not found:
                new_basis.append(None)
        else:
            new_basis.append(col_map.get(bi, None))

    basis = new_basis

    # Phase 2: build original objective row (we minimized, so c already adjusted)
    row0 = np.zeros(total_vars + 1, dtype=float)
    # c might have less columns now: original n plus slack/surplus
    # build full c_extended
    c_ext = np.zeros(total_vars, dtype=float)
    # map original coefficients
    for j in range(n):
        # find index of xj in keep_cols
        if j in col_map:
            idx = col_map[j]
            c_ext[idx] = c[j]
            
    row0[:-1] = -c_ext
    row0[-1] = 0.0

    # eliminate basic variables from row0
    for i_row in range(1, tableau.shape[0]):
        bcol = basis[i_row - 1]
        if bcol is None:
            continue
        coef = c_ext[bcol]
        if abs(coef) > tol:
            row0[:] += coef * tableau[i_row, :]

    # replace tableau[0]
    tableau[0, :] = row0

    # Simplex loop for Phase 2
    iter_count = 0
    while True:
        if iter_count > maxiter:
            status = "iteration_limit_phase2"
            break
        # entering for minimization: positive coefficient
        entering = choose_entering(tableau)
        if entering is None:
            status = "optimal"
            break
        leaving = choose_leaving(tableau, entering)
        if leaving is None:
            status = "unbounded"
            break
        # update basis
        basis[leaving - 1] = entering
        pivot(tableau, leaving, entering)
        iter_count += 1

    if status == "unbounded":
        return {"status": "unbounded", "x": None, "objective": None, "bv": None, "nbv": None}

    # Extract solution and separate BV/NBV
    x = np.zeros(n, dtype=float)
    bv_dict = {}
    nbv_dict = {}
    
    # Identify basic variables and their values
    basic_indices = set()
    for i_row in range(1, tableau.shape[0]):
        bcol = basis[i_row - 1]
        if bcol is not None:
            val = tableau[i_row, -1]
            var_name = phase2_var_names[bcol]
            bv_dict[var_name] = float(val)
            basic_indices.add(bcol)
            
            # If this basic column corresponds to an original variable
            orig_idx = keep_cols[bcol]
            if orig_idx < n:
                x[orig_idx] = val
    
    # Non-basic variables are those in phase2_var_names but not in basis
    for j in range(total_vars):
        if j not in basic_indices:
            var_name = phase2_var_names[j]
            nbv_dict[var_name] = 0.0

    # Compute original objective value using the original coefficients
    obj = float(c_orig.dot(x))

    return {
        "status": status,
        "x": x,
        "objective": obj,
        "bv": bv_dict,
        "nbv": nbv_dict
    }
