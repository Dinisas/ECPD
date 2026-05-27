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
% End of Q4 original section

% ═════════════════════════════════════════════════════════════════════════
%  Q4.5a — NON-REGULARIZED sparse solver: hard vs soft, effect of alpha
%
%  Shows that without Tikhonov regularisation the hard constraint can produce
%  poor conditioning / unexpected exitflags, motivating the switch to the
%  regularised solver with soft constraints.
% ═════════════════════════════════════════════════════════════════════════

% ── Reference (same as Q4.5 — above 55 °C) ───────────────────────────────
Dr_q5     = 55 - y_ss + 5;
y_max_q5  = 55;
y_max_inc_q5 = (y_max_q5 - y_ss) - Dr_q5;

% Feedforward for Dr_q5
M_q5   = [eye(n)-A, -B; C, 0];
b_q5   = [zeros(n,1); Dr_q5];
sol_q5 = M_q5 \ b_q5;
Dx_bar_q5 = sol_q5(1:n);
Du_bar_q5 = sol_q5(end);
lb_q5 = (-u_ss       - Du_bar_q5) * ones(H,1);
ub_q5 = ( 100-u_ss   - Du_bar_q5) * ones(H,1);
x0_q5 = x_ss + Dx0;
t_q5  = (0:N-1) * Ts;

fprintf('\n=== Q4.5a — NON-REGULARIZED: Safety constraint test ===\n')
fprintf('  Reference r = %.2f °C  (y_max = %.0f °C)\n', y_ss + Dr_q5, y_max_q5)

q5_cases = { 'Hard constraint',   0,   NaN  ; ...
             'Soft  \alpha=1',    1,   1    ; ...
             'Soft  \alpha=10',   1,   10   ; ...
             'Soft  \alpha=100',  1,   100  ; ...
             'Soft  \alpha=1000', 1,   1000 };
n_q5 = size(q5_cases, 1);

Y_q5a  = nan(n_q5, N);
U_q5a  = nan(n_q5, N);
EF_q5a = nan(n_q5, N);

for ci = 1:n_q5
    lbl       = q5_cases{ci,1};
    ctype     = q5_cases{ci,2};
    alpha_val = q5_cases{ci,3};
    fprintf('  Running: %s ...', lbl)

    x_ci      = nan(n, N+1);
    x_ci(:,1) = x0_q5;

    for k = 1:N
        y_k  = T1C(x_ci(:,k));
        Dx_k = x_ci(:,k) - x_ss;
        dx_k = Dx_k - Dx_bar_q5;

        if ctype == 0
            [du_k, ef_k] = mpc_solve_sparse( ...
                dx_k, H, R, A, B, C, lb_q5, ub_q5, y_max_inc_q5, 0);
        else
            [du_k, ef_k] = mpc_solve_sparse( ...
                dx_k, H, R, A, B, C, lb_q5, ub_q5, y_max_inc_q5, 1, alpha_val);
        end

        Du_k        = du_k + Du_bar_q5;
        u_k         = u_ss + Du_k;
        Y_q5a(ci,k) = y_k;
        U_q5a(ci,k) = u_k;
        EF_q5a(ci,k)= ef_k;
        x_ci(:,k+1) = h1(x_ci(:,k), u_k);
    end

    n_inf = sum(EF_q5a(ci,:) ~= 1);
    fprintf('  infeasible steps: %d / %d\n', n_inf, N)
end
fprintf('Q4.5a done.\n')

% ── Plot Q4.5a ────────────────────────────────────────────────────────────
colors_q5 = lines(n_q5);
figure('Units','normalized','Position',[0.05 0.05 0.55 0.42])
subplot(2,1,1), hold on, grid on
title(sprintf('Q4.5a — NON-REGULARIZED: Hard vs Soft  (r = %.1f °C, y_{max} = %.0f °C)', ...
              y_ss + Dr_q5, y_max_q5))
for ci = 1:n_q5
    plot(t_q5, Y_q5a(ci,:), 'Color', colors_q5(ci,:), 'LineWidth', 1.5, ...
         'DisplayName', q5_cases{ci,1})
end
yline(y_ss + Dr_q5, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Reference r')
yline(y_max_q5,     'r--', 'LineWidth', 2,   'DisplayName', 'y_{max} = 55°C')
xlabel('Time [s]'), ylabel('y [°C]')
legend('Location','best','Interpreter','tex')

subplot(2,1,2), hold on, grid on
for ci = 1:n_q5
    stairs(t_q5, U_q5a(ci,:), 'Color', colors_q5(ci,:), 'LineWidth', 1.5, ...
           'DisplayName', q5_cases{ci,1})
end
yline(100, 'r--', 'HandleVisibility','off')
xlabel('Time [s]'), ylabel('u [%]')
ylim([20 100])
legend('Location','best','Interpreter','tex')

% ═════════════════════════════════════════════════════════════════════════
%  Q4.5b — REGULARIZED sparse solver: hard vs soft, effect of alpha
%
%  Reference is set ABOVE 55 °C so the hard constraint becomes infeasible.
%  We then switch to soft constraints and compare alpha = 1, 10, 100, 1000.
% ═════════════════════════════════════════════════════════════════════════

% All setup variables (Dr_q5, y_max_inc_q5, Dx_bar_q5, lb_q5, ub_q5,
% x0_q5, t_q5, q5_cases, n_q5) are already defined in Q4.5a above.

fprintf('\n=== Q4.5b — REGULARIZED: Safety constraint test ===\n')
fprintf('  Reference r = %.2f °C  (y_max = %.0f °C)\n', y_ss + Dr_q5, y_max_q5)

% Storage
Y_q5b  = nan(n_q5, N);
U_q5b  = nan(n_q5, N);
EF_q5b = nan(n_q5, N);

% ── Sequential simulation loop ────────────────────────────────────────────
for ci = 1:n_q5
    lbl       = q5_cases{ci,1};
    ctype     = q5_cases{ci,2};
    alpha_val = q5_cases{ci,3};

    fprintf('  Running: %s ...', lbl)

    x_ci      = nan(n, N+1);
    x_ci(:,1) = x0_q5;

    for k = 1:N
        y_k  = T1C(x_ci(:,k));
        Dx_k = x_ci(:,k) - x_ss;
        dx_k = Dx_k - Dx_bar_q5;

        if ctype == 0
            [du_k, ef_k] = mpc_solve_sparse_regularized( ...
                dx_k, H, R, A, B, C, lb_q5, ub_q5, y_max_inc_q5, 0);
        else
            [du_k, ef_k] = mpc_solve_sparse_regularized( ...
                dx_k, H, R, A, B, C, lb_q5, ub_q5, y_max_inc_q5, 1, alpha_val);
        end

        Du_k         = du_k + Du_bar_q5;
        u_k          = u_ss + Du_k;
        Y_q5b(ci,k)  = y_k;
        U_q5b(ci,k)  = u_k;
        EF_q5b(ci,k) = ef_k;
        x_ci(:,k+1)  = h1(x_ci(:,k), u_k);
    end

    n_inf = sum(EF_q5b(ci,:) ~= 1);
    fprintf('  infeasible steps: %d / %d\n', n_inf, N)
end
fprintf('Q4.5b done.\n')

% ── Plot Q4.5b ────────────────────────────────────────────────────────────
figure('Units','normalized','Position',[0.42 0.05 0.55 0.42])
subplot(2,1,1), hold on, grid on
title(sprintf('Q4.5b — REGULARIZED: Hard vs Soft  (r = %.1f °C, y_{max} = %.0f °C)', ...
              y_ss + Dr_q5, y_max_q5))
for ci = 1:n_q5
    plot(t_q5, Y_q5b(ci,:), 'Color', colors_q5(ci,:), 'LineWidth', 1.5, ...
         'DisplayName', q5_cases{ci,1})
end
yline(y_ss + Dr_q5, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Reference r')
yline(y_max_q5,     'r--', 'LineWidth', 2,   'DisplayName', 'y_{max} = 55°C')
xlabel('Time [s]'), ylabel('y [°C]')
legend('Location','best','Interpreter','tex')

subplot(2,1,2), hold on, grid on
for ci = 1:n_q5
    stairs(t_q5, U_q5b(ci,:), 'Color', colors_q5(ci,:), 'LineWidth', 1.5, ...
           'DisplayName', q5_cases{ci,1})
end
yline(100, 'r--', 'HandleVisibility','off')
xlabel('Time [s]'), ylabel('u [%]')
ylim([20 100])
legend('Location','best','Interpreter','tex')

%--------------------------------------------------------------------------
% End of File