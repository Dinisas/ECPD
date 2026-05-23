function [u0, exitflag] = mpc_solve_sparse(x0, H, R, A, B, C, lb, ub, y_max_inc)
% MPC_SOLVE_SPARSE  Solve the MPC regulator using quadprog — SPARSE formulation.
%
%   u0 = mpc_solve_sparse(x0, H, R, A, B, C)          % unconstrained
%   u0 = mpc_solve_sparse(x0, H, R, A, B, C, lb, ub)  % control bounds
%
%   Solves the same finite-horizon quadratic MPC problem as mpc_solve, but
%   using the SPARSE formulation, i.e. both state and control variables are
%   collocated as optimization variables:
%
%       z = [ X ]  =  [ xhat(0); xhat(1); ... ; xhat(H);
%             U ]       uhat(0) ; uhat(1); ... ; uhat(H-1) ]
%
%   with dimensions:
%       X  :  (H+1)*n  x 1
%       U  :  H        x 1
%       z  :  (H+1)*n + H  x 1
%
%   The cost is:
%
%       J = sum_{i=0}^{H-1} yhat(i+1)^2 + R*uhat(i)^2
%         = X' * Qtilde * X  +  U' * Rtilde * U
%
%   with Q_stage = C'*C (output penalty via Q = C'C),
%        Qtilde  = blkdiag( 0_n , Q_stage, ..., Q_stage )   [(H+1) blocks]
%        Rtilde  = R * I_H
%
%   Translated to quadprog form:
%       F = 2 * blkdiag(Qtilde, Rtilde),   f = 0
%
%   The system dynamics are imposed as equality constraints:
%
%       xhat(0)     = x0                      (initial condition)
%       xhat(i+1)   = A*xhat(i) + B*uhat(i)  (model, i = 0,...,H-1)
%
%   giving  Aeq * z = beq  (see code below).
%
%   Control bounds lb, ub are applied only to U (the state is unconstrained).
%
%   Inputs
%     x0  – n×1 current state (incremental: Dx at current time)
%     H   – prediction horizon (positive integer)
%     R   – positive scalar control weight
%     A   – n×n state matrix
%     B   – n×1 input matrix  (scalar input assumed)
%     C   – 1×n output matrix (scalar output assumed)
%     lb  – H×1 lower bound on DU (pass [] for unconstrained)
%     ub  – H×1 upper bound on DU (pass [] for unconstrained)
%
%   Output
%     u0  – scalar, first optimal control action (receding horizon)
%
%   Note: produces identical results to mpc_solve (dense formulation).
%   The sparse formulation is generally preferable when output/state
%   constraints are needed, since the state trajectory X is directly
%   available as an optimisation variable.

%% ── Defaults ─────────────────────────────────────────────────────────────
if nargin < 7,  lb = []; end
if nargin < 8,  ub = []; end
if nargin < 9,  y_max_inc = []; end     % empty = no output constraint

use_output_constraint = ~isempty(y_max_inc);

%% ── Dimensions ───────────────────────────────────────────────────────────
n   = size(A, 1);
N_x = (H+1) * n;   % number of state  optimisation variables
N_u = H;            % number of control optimisation variables
N_z = N_x + N_u;   % total size of z

%% ── Cost matrices ────────────────────────────────────────────────────────
% Stage output cost: Q_stage = C'*C  (n×n positive semidefinite)
Q_stage = C' * C;

% Qtilde = blkdiag(0_n, Q_stage, ..., Q_stage)  — (H+1) diagonal blocks
%   first block is 0 because xhat(0) does not appear in the cost
Qtilde = blkdiag(zeros(n), kron(eye(H), Q_stage));

% Rtilde = R * I_H
Rtilde = R * eye(H);

% quadprog minimises  1/2 * z'*F*z + f'*z
%   J = X'*Qtilde*X + U'*Rtilde*U
%     = (1/2)*z'* 2*blkdiag(Qtilde,Rtilde) *z + 0'*z
F = 2 * blkdiag(Qtilde, Rtilde);
f = zeros(N_z, 1);

%% ── Equality constraints  (dynamics + initial condition) ─────────────────
% Layout of z (column vector):
%   z(      1 :       n) = xhat(0)
%   z(    n+1 :     2*n) = xhat(1)
%   ...
%   z(  H*n+1 : (H+1)*n) = xhat(H)
%   z(N_x + 1 : N_x + H) = [uhat(0); ...; uhat(H-1)]
%
% Constraint rows:
%   rows    1 :   n            →  xhat(0) = x0
%   rows  n+1 : 2*n            →  xhat(1) - A*xhat(0) - B*uhat(0) = 0
%   ...
%   rows H*n+1 : (H+1)*n       →  xhat(H) - A*xhat(H-1) - B*uhat(H-1) = 0

N_eq = (H+1) * n;
Aeq  = zeros(N_eq, N_z);
beq  = zeros(N_eq, 1);

% --- Initial condition block: xhat(0) = x0 --------------------------------
Aeq(1:n, 1:n) = eye(n);
beq(1:n)      = x0;

% --- Dynamics blocks: xhat(i+1) - A*xhat(i) - B*uhat(i) = 0 --------------
for i = 0:H-1
    r1 = n + i*n + 1;       % first row for xhat(i+1) constraint
    r2 = n + (i+1)*n;       % last  row for xhat(i+1) constraint

    % coefficient of xhat(i+1): +I
    c1_xi1 = (i+1)*n + 1;
    c2_xi1 = (i+2)*n;
    Aeq(r1:r2, c1_xi1:c2_xi1) = eye(n);

    % coefficient of xhat(i):   -A
    c1_xi = i*n + 1;
    c2_xi = (i+1)*n;
    Aeq(r1:r2, c1_xi:c2_xi) = -A;

    % coefficient of uhat(i):   -B  (scalar control => single column)
    c_ui = N_x + i + 1;
    Aeq(r1:r2, c_ui) = -B;
end

%% ── Bounds on z ──────────────────────────────────────────────────────────
% State variables are unconstrained; bounds apply only to U.
if ~isempty(lb) || ~isempty(ub)
    if isempty(lb), lb = -inf(N_u, 1); end
    if isempty(ub), ub =  inf(N_u, 1); end
    lb_z = [-inf(N_x, 1); lb];
    ub_z = [ inf(N_x, 1); ub];
else
    lb_z = [];
    ub_z = [];
end


%% Constraint: W*U <= y_max_inc*ones(H,1) - Pi*x0
if use_output_constraint
    % C_til maps X = [xhat(0); xhat(1); ...; xhat(H)] to [yhat(1); ...; yhat(H)]
    % xhat(0) does not appear in the cost -> prepend zero block
    C_til  = kron(eye(H), C);            % H × H*n  (outputs yhat(1)..yhat(H))
    C_til  = [zeros(H, n), C_til];       % H × (H+1)*n  (prepend zero for xhat(0))

    % Full inequality: [C_til, 0_{H×H}] * z <= y_max_inc * ones(H,1)
    A_ineq = [C_til, zeros(H, N_u)];     % H × N_z
    b_ineq = y_max_inc * ones(H, 1);     % H × 1
else
    A_ineq = [];
    b_ineq = [];
end

%% ── Solve with quadprog ──────────────────────────────────────────────────
opts  = optimoptions('quadprog', 'Display', 'off');
[U_opt, ~, exitflag] = quadprog(F, f, A_ineq, b_ineq, Aeq, beq, lb_z, ub_z, [], opts);

%% ── Receding horizon: return first control action ────────────────────────
% U starts at index N_x + 1 in z
u0 = U_opt(N_x + 1);

end
