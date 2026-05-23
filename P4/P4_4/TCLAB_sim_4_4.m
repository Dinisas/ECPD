% Simulation of the TCLab linear model previously identified — Q4
%
% The identified model is purely incremental:
%       Dx(k+1) = A*Dx(k) + B*Du(k) + Ke*e(k)
%       Dy(k)   = C*Dx(k) + e(k)
% The simulator wraps it in absolute coordinates via the bias terms c1, c2.
%
% Q4 — Tracking a reference Dr with feedforward control + perturbation study
%
% Afonso Botelho and J. Miranda Lemos, IST, May 2023
%__________________________________________________________________________
clear all
close all
clc

% ── Load model ────────────────────────────────────────────────────────────
load('singleheater_model2D.mat','A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n = size(A,1);

% Noise (set to 0 for clean tuning; turn on for final plots if desired)
e_std = sqrt(e_var);
%e_std = 0;

% ── Simulation parameters ─────────────────────────────────────────────────
T = 400;        % Experiment duration [s]
N = T/Ts;       % Number of samples

% ── MPC tuning (fixed from Q2/Q3) ─────────────────────────────────────────
H = 50;        % prediction horizon (fixed from Q2)
R = 0.04;       % control weight     (fixed from Q3)

% ── Toggles ──────────────────────────────────────────────────────────────
% perturb_amount: constant additive input disturbance applied to the plant.
%   - The plant feels:   B*(u + d)   where d = perturb_amount  [% of heater]
%   - The MPC's model assumes d = 0.
%   - This mimics the guide's "10% increase in c1" (ambient drift /
%     unmodelled effects) without requiring c1 to be non-zero.
% Set to 0 to disable the disturbance.
perturb_amount = 0;     % constant heater offset the MPC is blind to [%]

% initial offset of the output from equilibrium (just for the transient)
offset = 0;            % desired Dy(1) = offset [°C]

% ═════════════════════════════════════════════════════════════════════════
%  Q4 — REFERENCE TRACKING WITH FEEDFORWARD
% ═════════════════════════════════════════════════════════════════════════

% ── Reference (incremental) ───────────────────────────────────────────────
Dr = 15;         % desired output increment [°C]  -> r = y_ss + Dr

% HARD constraint δy_hat​(i)≤55−y_bar ​−Δr
y_max = 55; 
y_max_inc = (y_max-y_ss)-Dr; 

mode = 1; % 0 is dense ; 1 is sparse
const_type = 1; %0 is hard, 1 is soft

% ── Q4.A — Feedforward: solve steady-state equations ─────────────────────
% Find (Dx_bar, Du_bar) such that the system is in equilibrium at Dy = Dr.
%       (I - A) Dx_bar = B Du_bar
%       C Dx_bar       = Dr
M   = [eye(n)-A, -B; C, 0];
b   = [zeros(n,1); Dr];
sol = M \ b;

Dx_bar = sol(1:n);
Du_bar = sol(end);

% ── Q4.B — Shifted control limits ────────────────────────────────────────
% Original limits: -u_ss <= Du_hat <= 100 - u_ss
% After change of variables du_hat = Du_hat - Du_bar:
%       -u_ss - Du_bar <= du_hat <= 100 - u_ss - Du_bar
lb = (-u_ss       - Du_bar) * ones(H,1);
ub = ( 100 - u_ss - Du_bar) * ones(H,1);

% ── Build the simulator handles ──────────────────────────────────────────
% c1, c2 are bias terms that make the absolute-variable simulator equivalent
% to the incremental model around (x_ss, u_ss, y_ss). By construction they
% come out essentially zero — that's normal.
x_ss = [eye(n)-A; C] \ [B*u_ss; y_ss];
c1   = (eye(n)-A)*x_ss - B*u_ss;
c2   = y_ss - C*x_ss;

% Constant additive input disturbance (computed ONCE, applied every step).
% This is the disturbance d in the model
%       Dx(k+1) = A Dx(k) + B (Du(k) + d) + ...
% i.e. an extra heater offset that the MPC's model knows nothing about.
d_offset = B * perturb_amount;     % n×1 vector, computed ONCE

h1  = @(x,u) A*x + B*u + Ke*e_std*randn + c1 + d_offset;   % apply control
T1C = @(x)   C*x + e_std*randn + c2;                       % read temperature

% ── Initial condition: start away from equilibrium by Dy0 = offset ───────
% Solve for the minimum-norm Dx0 such that C*Dx0 = offset.
if offset ~= 0
    Dx0 = C' / (C*C') * offset;
else
    Dx0 = zeros(n,1);
end

x      = nan(n, N+1);
x(:,1) = x_ss + Dx0;

% Initial conditions (start at ambient temperature, i.e. equilibrium for u = 0)
%Dx0Dy0 = [eye(n)-A, zeros(n,1); C, -1]\[-B*u_ss; 0];
%Dx0 = Dx0Dy0(1:n);
% ...
%x(:,1) = Dx0 + x_ss;

% ── Initialize signals ───────────────────────────────────────────────────
t  = nan(1, N);
y  = nan(1, N);
Dy = nan(1, N);
Du = nan(1, N);
Dx = nan(n, N);
u  = nan(1, N);
exitflag = nan(1, N); % For question 5 

% ── Diagnostic prints ────────────────────────────────────────────────────
fprintf('--- Q4 diagnostics ---\n')
fprintf('  c1 norm        = %.3e   (should be ~0 by construction)\n', norm(c1))
fprintf('  c2             = %.3e   (should be ~0 by construction)\n', c2)
fprintf('  x(:,1)         = [%s]\n', sprintf('%.3f  ', x(:,1)))
fprintf('  y(1) predicted = %.3f °C\n', C*x(:,1) + c2)
fprintf('  Dx_bar         = [%s]\n', sprintf('%.4f  ', Dx_bar))
fprintf('  Du_bar         = %.4f %%\n', Du_bar)
fprintf('  reference r    = %.3f °C\n', y_ss + Dr)
fprintf('  perturb_amount = %.2f %%   (d_offset norm = %.4e)\n', ...
        perturb_amount, norm(d_offset))
fprintf('-----------------------\n')

% ═════════════════════════════════════════════════════════════════════════
%  Closed-loop simulation
% ═════════════════════════════════════════════════════════════════════════
fprintf('Running Q4 simulation (Dr = %.2f, perturb = %.2f%%) ...\n', ...
        Dr, perturb_amount)
for k = 1:N
    t(k)    = (k-1)*Ts;

    % Sense
    y(:,k)  = T1C(x(:,k));
    Dy(:,k) = y(:,k) - y_ss;
    Dx(:,k) = x(:,k) - x_ss;

    % ── Q4.C — Change of variables and regulator call ───────────────────
    dx_k    = Dx(:,k) - Dx_bar;                          % shift state
    %du_k    = mpc_solve(dx_k, H, R, A, B, C, lb, ub);    % regulator in shifted coords
    %[du_k, exitflag(k)] = mpc_solve(dx_k, H, R, A, B, C, lb, ub, y_max_inc);

    if mode == 0
      [du_k, exitflag(k)] = mpc_solve(dx_k, H, R, A, B, C, lb, ub, y_max_inc);
    elseif mode == 1
      [du_k, exitflag(k)] = mpc_solve_sparse_regularized(dx_k, H, R, A, B, C, lb, ub, y_max_inc,const_type);
    end

    Du(:,k) = du_k + Du_bar;                             % reconstruct increment
    u(:,k)  = u_ss + Du(:,k);                            % absolute control

    % Act
    x(:,k+1) = h1(x(:,k), u(:,k));
end
fprintf(' Done.\n');
fprintf('Infeasible steps: %d / %d\n', sum(exitflag ~= 1), N)

% ── Final report ─────────────────────────────────────────────────────────
fprintf('Final y      = %.3f °C\n', y(end))
fprintf('Reference r  = %.3f °C\n', y_ss + Dr)
fprintf('Offset       = %+.3f °C\n', y(end) - (y_ss + Dr))

% ═════════════════════════════════════════════════════════════════════════
%  Plots
% ═════════════════════════════════════════════════════════════════════════

% ── Absolute variables ──────────────────────────────────────────────────
figure('Units','normalized','Position',[0.2 0.5 0.3 0.4])
subplot(2,1,1), hold on, grid on
title(sprintf('Q4 — Absolute  (\\Delta r = %.1f, d = %.1f%%)', Dr, perturb_amount))
plot(t, y, '.', 'MarkerSize', 5)
yl_r = yline(y_ss + Dr, 'g--', 'LineWidth', 1.5);
yl_y = yline(y_ss,      'k--');
xlabel('Time [s]'), ylabel('y [°C]')
legend([yl_r, yl_y], {'$r = \bar{y} + \Delta r$', '$\bar{y}$'}, ...
       'Interpreter','latex','Location','best')

subplot(2,1,2), hold on, grid on
stairs(t, u, 'LineWidth', 2)
yl_u = yline(u_ss, 'k--');
yline(0,  'r--')
yline(100,'r--')
xlabel('Time [s]'), ylabel('u [%]')
legend(yl_u, '$\bar{u}$', 'Interpreter','latex','Location','best')

% ── Incremental variables ───────────────────────────────────────────────
figure('Units','normalized','Position',[0.5 0.5 0.3 0.4])
subplot(2,1,1), hold on, grid on
title(sprintf('Q4 — Incremental  (\\Delta r = %.1f, d = %.1f%%)', Dr, perturb_amount))
plot(t, Dy, '.', 'MarkerSize', 5)
yline(Dr, 'g--', 'LineWidth', 1.5)
yline(0,  'k--')
xlabel('Time [s]'), ylabel('\Delta y [°C]')

subplot(2,1,2), hold on, grid on
stairs(t, Du, 'LineWidth', 2)
yline(-u_ss,     'r--')
yline(100-u_ss,  'r--')
xlabel('Time [s]'), ylabel('\Delta u [%]')

%--------------------------------------------------------------------------
% End of File