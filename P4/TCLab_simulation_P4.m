% Simulation of the TCLab linear model previously identified
%
% The identified model is purely incremental:
%       Dx(k+1) = A*Dx(k) + B*Du(k) + Ke*e(k)
%       Dy(k)   = C*Dx(k) + e(k)
% so this simulation works directly in incremental coordinates and only
% reconstructs absolute (y, u) for plotting.
%
% Q2: Unconstrained closed-loop MPC — study effect of H (fixed R) and R (fixed H)
% Q3: Constrained closed-loop MPC   — add control limits [0,100]%, tune R for saturation
%
% Afonso Botelho and J. Miranda Lemos, IST, May 2023
%__________________________________________________________________________

clear all
close all
clc

% ── Paths ─────────────────────────────────────────────────────────────────
base = fileparts(mfilename('fullpath'));   % folder where this script lives
addpath(fullfile(base, '../P3'));          % singleheater_model.mat
addpath(fullfile(base, '../P4'));          % mpc_solve_sparse_regularized.m

% ── Load model ────────────────────────────────────────────────────────────
load('singleheater_model_n2.mat','A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n = size(A,1);

% Noise: keep at 0 during MPC tuning (ideal conditions)
e_std = 0;

% ── Simulation parameters ─────────────────────────────────────────────────
T = 1000;       % Experiment duration [s]
N = T/Ts;       % Number of samples

% ── Initial incremental state ─────────────────────────────────────────────
% Start the simulation exactly 10 degrees below the steady-state equilibrium.
% This ensures we test the controller within the model's valid linear region,
% avoiding the "ambient blind spot" while still forcing a transient response.
Dy_start = -26; %20ºC
Dx0 = pinv(C) * Dy_start;

%% ═══════════════════════════════════════════════════════════════════════
%  Q2 — UNCONSTRAINED MPC
% ═══════════════════════════════════════════════════════════════════════

%% ── Q2 Study 1: Effect of H (R fixed) ───────────────────────────────────
R_fixed = 1;
H_list  = [3, 5, 10, 20, 50, 75, 90, 100];      % H >= n = 3 (model order)

colors_H = lines(length(H_list));

figure('Units','normalized','Position',[0.05 0.5 0.55 0.4])
subplot(2,1,1), hold on, grid on
title(sprintf('Q2 — Effect of H on \\Deltay  (R = %.1f fixed)', R_fixed))
ylabel('\Delta y [°C]'), xlabel('Time [s]')
yline(0,'k--','LineWidth',1)
subplot(2,1,2), hold on, grid on
title('Corresponding \Deltau  (unconstrained)')
ylabel('\Delta u [%]'), xlabel('Time [s]')

for i = 1:length(H_list)
    H_i = H_list(i);
    fprintf('Q2 Study 1 — H=%d, R=%.1f ...\n', H_i, R_fixed)

    t  = nan(1,N);
    Dy = nan(1,N);
    Du = nan(1,N);
    Dx = nan(n,N+1);
    Dx(:,1) = Dx0;

    for k = 1:N
        t(k)      = (k-1)*Ts;
        Dy(k)     = C*Dx(:,k) + e_std*randn;
        Du(k)     = mpc_solve_sparse_regularized(Dx(:,k), H_i, R_fixed, A, B, C);   % unconstrained
        Dx(:,k+1) = A*Dx(:,k) + B*Du(k) + Ke*e_std*randn;
    end

    subplot(2,1,1)
    plot(t, Dy, 'Color', colors_H(i,:), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('H = %d', H_i))
    subplot(2,1,2)
    stairs(t, Du, 'Color', colors_H(i,:), 'LineWidth', 1.5, ...
           'DisplayName', sprintf('H = %d', H_i))
end
subplot(2,1,1), legend('Location','best')
subplot(2,1,2), legend('Location','best')

%% ── Q2 Study 2: Effect of R (H fixed at chosen value) ───────────────────
H_chosen = 50;      % <-- update after analysing Study 1
R_list   = [0.01, 0.025, 0.04, 0.05, 0.1, 1];

colors_R = lines(length(R_list));

figure('Units','normalized','Position',[0.55 0.5 0.55 0.4])
subplot(2,1,1), hold on, grid on
title(sprintf('Q2 — Effect of R on \\Deltay  (H = %d fixed)', H_chosen))
ylabel('\Delta y [°C]'), xlabel('Time [s]')
yline(0,'k--','LineWidth',1)
subplot(2,1,2), hold on, grid on
title('Corresponding \Deltau  (unconstrained)')
ylabel('\Delta u [%]'), xlabel('Time [s]')

for i = 1:length(R_list)
    R_i = R_list(i);
    fprintf('Q2 Study 2 — H=%d, R=%.3f ...\n', H_chosen, R_i)

    t  = nan(1,N);
    Dy = nan(1,N);
    Du = nan(1,N);
    Dx = nan(n,N+1);
    Dx(:,1) = Dx0;

    for k = 1:N
        t(k)      = (k-1)*Ts;
        Dy(k)     = C*Dx(:,k) + e_std*randn;
        Du(k)     = mpc_solve_sparse_regularized(Dx(:,k), H_chosen, R_i, A, B, C);   % unconstrained
        Dx(:,k+1) = A*Dx(:,k) + B*Du(k) + Ke*e_std*randn;
    end

    subplot(2,1,1)
    plot(t, Dy, 'Color', colors_R(i,:), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('R = %.3f', R_i))
    subplot(2,1,2)
    stairs(t, Du, 'Color', colors_R(i,:), 'LineWidth', 1.5, ...
           'DisplayName', sprintf('R = %.3f', R_i))
end
subplot(2,1,1), legend('Location','best')
subplot(2,1,2), legend('Location','best')

%% ═══════════════════════════════════════════════════════════════════════
%  Q3 — CONSTRAINED MPC
% ═══════════════════════════════════════════════════════════════════════

% Control bounds in incremental form (fixed, computed once before the loop)
lb = -u_ss       * ones(H_chosen, 1);   % Du >= -u_ss  =>  u >= 0
ub = (100-u_ss)  * ones(H_chosen, 1);   % Du <= 100-u_ss  =>  u <= 100

R_Q3 = 0.04;    % <-- tune until saturation appears
fprintf('Q3 — H=%d, R=%.3f (constrained) ...\n', H_chosen, R_Q3)

t  = nan(1,N);
Dy = nan(1,N);
Du = nan(1,N);
Dx = nan(n,N+1);
Dx(:,1) = Dx0;

for k = 1:N
    t(k)      = (k-1)*Ts;
    Dy(k)     = C*Dx(:,k) + e_std*randn;
    
    % Note: If you experience numeric chattering with H=100 and dense solver,
    % you can swap this to mpc_solve_sparse_regularized(...)
    Du(k)     = mpc_solve_sparse_regularized(Dx(:,k), H_chosen, R_Q3, A, B, C, lb, ub);   % constrained
    
    Dx(:,k+1) = A*Dx(:,k) + B*Du(k) + Ke*e_std*randn;
end

% Reconstruct absolute signals for plotting
y = Dy + y_ss;
u = Du + u_ss;

% Plot absolute variables
figure('Units','normalized','Position',[0.2 0.05 0.3 0.4])
subplot(2,1,1), hold on, grid on
title(sprintf('Q3 — Absolute  (H=%d, R=%.3f, constrained)', H_chosen, R_Q3))
plot(t, y, '.', 'MarkerSize', 5)
yl = yline(y_ss,'k--');
xlabel('Time [s]'), ylabel('y [°C]')
legend(yl,'$\bar{y}$','Interpreter','latex','Location','best')
subplot(2,1,2), hold on, grid on
stairs(t, u, 'LineWidth', 2)
yl = yline(u_ss,'k--');
yline(0,  'r--')
yline(100,'r--')
xlabel('Time [s]'), ylabel('u [%]')
legend(yl,'$\bar{u}$','Interpreter','latex','Location','best')

% Plot incremental variables
figure('Units','normalized','Position',[0.5 0.05 0.3 0.4])
subplot(2,1,1), hold on, grid on
title(sprintf('Q3 — Incremental  (H=%d, R=%.3f, constrained)', H_chosen, R_Q3))
plot(t, Dy, '.', 'MarkerSize', 5)
yline(0,'k--')
xlabel('Time [s]'), ylabel('\Delta y [°C]')
subplot(2,1,2), hold on, grid on
stairs(t, Du, 'LineWidth', 2)
yline(-u_ss,     'r--')
yline(100-u_ss,  'r--')
xlabel('Time [s]'), ylabel('\Delta u [%]')

%--------------------------------------------------------------------------
% End of File