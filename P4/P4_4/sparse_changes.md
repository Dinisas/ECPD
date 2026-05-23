# Changes from Original to Final mpc_solve_sparse

## 1. Tikhonov Regularisation on Cost Matrix

**Original:**
```matlab
F = 2 * blkdiag(Qtilde, Rtilde);
```

**Final:**
```matlab
epsilon = 1e-6;
F = 2 * blkdiag(Qtilde + epsilon*eye((H+1)*n), Rtilde);
```

**Why:** Q_stage = C'*C has rank 1, making Qtilde rank-deficient with many zero
eigenvalues. This causes the KKT system quadprog must invert to be numerically
singular, producing erratic bang-bang control. Adding epsilon*I makes F strictly
positive definite, stabilising the solve with negligible effect on the solution (O(1e-6)).

---

## 2. quadprog Algorithm and Tolerances

**Original:**
```matlab
opts = optimoptions('quadprog', 'Display', 'off');
```

**Final:**
```matlab
opts = optimoptions('quadprog',            ...
    'Algorithm',           'interior-point-convex', ...
    'OptimalityTolerance', 1e-9,           ...
    'ConstraintTolerance', 1e-9,           ...
    'StepTolerance',       1e-12,          ...
    'Display',             'off');
```

**Why:** The default quadprog algorithm (active-set) is not designed for large
equality-constrained QPs. interior-point-convex solves the KKT system directly
and handles the (H+1)*n equality constraints reliably. Tighter tolerances ensure
the dynamics equality constraints are satisfied to high precision.

---

## 3. Loop Variable Renamed

**Original:**
```matlab
for i = 0:H-1
```

**Final:**
```matlab
for ii = 0:H-1
```

**Why:** MATLAB reserves `i` and `j` for the imaginary unit. The assignment
itself flags this in Appendix 1, rule 5f.

---

## Summary

| | Original | Final |
|---|---|---|
| Cost matrix F | Semidefinite, may be singular | Strictly PD via epsilon*I |
| quadprog algorithm | Default (active-set) | interior-point-convex |
| Tolerances | Default (1e-8) | Tightened (1e-9 / 1e-12) |
| Loop variables | i, j | ii |
| Result | Erratic bang-bang control | Matches dense formulation |
