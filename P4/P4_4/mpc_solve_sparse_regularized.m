function [u0, exitflag] = mpc_solve_sparse_regularized(x0, H, R, A, B, C, lb, ub, y_max_inc, const_type, alpha)
% MPC_SOLVE_SPARSE_REGULARIZED  Sparse MPC via quadprog with Tikhonov regularisation.
% Solves the same problem as mpc_solve (dense) but keeping state variables
% as explicit optimisation variables, with dynamics as equality constraints.
%
% const_type = 0 : hard output constraint
% const_type = 1 : soft output constraint (slack variables penalised by alpha)

if nargin < 7,   lb         = []; end
if nargin < 8,   ub         = []; end
if nargin < 9,   y_max_inc  = []; end
if nargin < 10,  const_type = 0;  end  % hard by default
if nargin < 11,  alpha      = 1e6; end % large default -> near-hard behaviour

use_output_constraint = ~isempty(y_max_inc);

%% Dimensions
n   = size(A, 1);
N_x = (H+1) * n;
N_u = H;
N_eta = N_u; % for y(i) 1x1
N_z = N_x + N_u;
N_z_soft = N_z + N_eta;

%% Cost  --  F = 2*blkdiag(Qtilde, Rtilde) + epsilon*I
% Q_stage = C'*C has rank 1, so Qtilde is rank-deficient (many zero
% eigenvalues).  The epsilon*I regularisation makes F strictly positive
% definite so that the quadprog KKT system is well-conditioned.
Q_stage = C' * C;
Qtilde  = blkdiag(zeros(n), kron(eye(H), Q_stage));
Rtilde  = R * eye(H);
epsilon = 1e-6;

% More principled: only regularise the ill-conditioned state block
F = 2 * blkdiag(Qtilde + epsilon*eye((H+1)*n), Rtilde);
f = zeros(N_z, 1);

% If using soft constraints (const_type==1) we have augmented variables z_soft
% that include eta (N_eta = H). Build F_soft and f_soft compatible with z_soft.
if const_type == 1
    % Extend F to include quadratic penalty on eta (use alpha on diagonal).
    % Place large weight alpha on slack variables in F so objective penalises them

    F = blkdiag(F, alpha * eye(N_eta));
    f = [f; zeros(N_eta,1)];
end

%% Equality constraints: dynamics + initial condition
N_eq = (H+1) * n;
Aeq  = zeros(N_eq, N_z);
beq  = zeros(N_eq, 1);

% Initial condition: xhat(0) = x0
Aeq(1:n, 1:n) = eye(n);
beq(1:n)      = x0;

% Dynamics: xhat(ii+1) - A*xhat(ii) - B*uhat(ii) = 0
for ii = 0:H-1
    r1 = n + ii*n + 1;
    r2 = n + (ii+1)*n;
    Aeq(r1:r2, (ii+1)*n+1:(ii+2)*n) =  eye(n);   % +I on xhat(ii+1)
    Aeq(r1:r2,     ii*n+1:(ii+1)*n) = -A;         % -A on xhat(ii)
    Aeq(r1:r2,            N_x+ii+1) = -B;         % -B on uhat(ii)
end

% If using soft constraints, Aeq must match extended decision vector size
% by adding zeros for the slack variables (eta). Do not modify RHS.
if const_type == 1
    % Extend Aeq with N_eta zero columns
    Aeq = [Aeq, zeros(N_eq, N_eta)];
end

%% Control bounds
if const_type==0
    if ~isempty(lb) || ~isempty(ub)
        if isempty(lb), lb = -inf(N_u, 1); end
        if isempty(ub), ub =  inf(N_u, 1); end
        lb_z = [-inf(N_x, 1); lb];
        ub_z = [ inf(N_x, 1); ub];
    else
        lb_z = [];
        ub_z = [];
    end
end 
if const_type==1 % modified to include eta
    if ~isempty(lb) || ~isempty(ub)
        if isempty(lb), lb = -inf(N_u, 1); end
        if isempty(ub), ub =  inf(N_u, 1); end
        lb_z = [-inf(N_x, 1); lb; ones(N_eta,1)*0.0001];% eta >=0
        ub_z = [ inf(N_x, 1); ub; inf(N_eta,1)]; 
    else
        lb_z = [];
        ub_z = [];
    end
end


%% Hard Output constraint: yhat(i) <= y_max_inc, i = 1,...,H
if use_output_constraint
    if const_type==0 %HARD
        C_til  = [zeros(H, n), kron(eye(H), C)];
        A_ineq = [C_til, zeros(H, N_u)];
        b_ineq = y_max_inc * ones(H, 1);
    end
    if const_type==1 %SOFT
        C_til  = [zeros(H, n), kron(eye(H), C)];
        A_ineq = [C_til, zeros(H, N_u), -eye(N_eta)];% include soft constraint
        b_ineq = y_max_inc * ones(H, 1);
    end
else
    A_ineq = [];
    b_ineq = [];
       
end

%% Solve
opts = optimoptions('quadprog',           ...
    'Algorithm',           'interior-point-convex', ...
    'OptimalityTolerance', 1e-9,          ...
    'ConstraintTolerance', 1e-9,          ...
    'StepTolerance',       1e-12,         ...
    'Display',             'off');

[U_opt, ~, exitflag] = quadprog(F, f, A_ineq, b_ineq, Aeq, beq, lb_z, ub_z, [], opts);

%% Receding horizon: return first control action
u0 = U_opt(N_x + 1);

end