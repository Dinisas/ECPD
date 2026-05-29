% Closed-loop experiment for data collection in TCLab -- P5
%
% Applies the Kalman filter and MPC designed in P4 to the REAL plant.
% Same specification as P4 Question 7: reference commutes 50 -> 40 -> 60
% -> 45 degC, with feedforward + disturbance estimation (zero-offset MPC).
%
% Recommended workflow: test the Kalman filter first with an open-loop
% profile (set USE_MPC = false), then enable the MPC (USE_MPC = true).
%
% Functions called: tclab, mpc_solve (or mpc_solve_sparse_regularized).
%
% J. Miranda Lemos and Afonso Botelho, IST, May 2023  --  P5 version
%__________________________________________________________________________

% Initialization
clear all
close all
clc

% Add required folders to path
base = fileparts(mfilename('fullpath'));
addpath(fullfile(base, '..', 'TCLab_software_test'));
addpath(fullfile(base, '..', 'P4'));

tclab;

% Load model
load(fullfile(base,'..','P4','P4_4','singleheater_model2D.mat'),'A','B','C','Ke','e_var','y_ss','u_ss','Ts');
n = size(A,1);

%% Experiment parameters ===============================================
% Same specification as P4 Q7: four reference levels, held for seg_time each.
%r_levels = [50 40 65 45];           % absolute reference [degC]
%r_levels = [50];
%seg_time = 400;                     % seconds held at each level
r_levels = [50 40 65 45 30 80 50 65 35 20 50];           % absolute reference [degC]
seg_time = 600;                     % seconds held at each level
%r_levels = [ 50 60 70  80  90 30 60 30 70 30 80 30 90];           % absolute reference [degC]
%seg_time = 400;                     % seconds held at each level
T = seg_time*numel(r_levels);       % experiment duration [s]
N = round(T/Ts);                    % number of samples to collect

%% Controller / observer options =======================================
USE_MPC    = true;   % false = open-loop test of the Kalman filter only
mode       = 1;      % 0 = dense mpc_solve, 1 = sparse mpc_solve_regularized
const_type = 1;      % 0 = hard output constraint, 1 = soft

% MPC tuning (fixed from P4)
H = 50;              % prediction horizon
R = 0.04;            % control weight

% Safety constraint
y_max  = 55;         % hard safety cap [degC]
margin = 0.5;        % clamp reference this far BELOW the cap [degC]

%% Kalman filter design ================================================
Ad = [A,            B;
      zeros(1,n),   1];
Bd = [B; 0];
Cd = [C, 0];

QE  = Ke * e_var * Ke';   % process-noise covariance (identified)
RE  = e_var;              % measurement-noise covariance (identified)
deltaE = 0.1;               % disturbance-state covariance -- TUNING KNOB

QEd = [QE,          zeros(n,1);
       zeros(1,n),  deltaE];

L = dlqe(Ad, eye(n+1), Cd, QEd, RE);   % Kalman gain

% Initial conditions (start at ambient temperature, i.e. equilibrium for u = 0)
Dx0Dy0 = [eye(n)-A, zeros(n,1); C, -1]\[-B*u_ss; 0];
Dx0 = Dx0Dy0(1:n); % to initialize filter

%% Reference =============================================================
% Build the absolute reference signal r(k) with four 400 s segments.
r = zeros(1,N);
for s = 1:numel(r_levels)
    idx = (1:N) > (s-1)*seg_time/Ts & (1:N) <= s*seg_time/Ts;
    r(idx) = r_levels(s);
end
Dr = r - y_ss;                       % incremental reference

% Steady-state matrix (constant -- reused every step in the loop)
Mss = [eye(n)-A, -B;
       C,         0];

% Real-time plot flag. If true, plots the input and measured temperature in
% real time. If false, only plots at the end of the experiment and instead
% prints the results in the command window.
rt_plot = true;

% Initialize figure and signals
if rt_plot
    figure
    drawnow;
end
t        = nan(1,N);
u        = zeros(1,N);
y        = zeros(1,N);
Dy       = nan(1,N);
Du       = zeros(1,N);
xd_est   = nan(n+1,N+1);
xd_est(:,1) = [Dx0; 0];              % Kalman filter initialization
exitflag = nan(1,N);

% String with date for saving results
timestr = char(datetime('now','Format','yyMMdd_HHmmSS'));

% Signals the start of the experiment by lighting the LED
led(1)
disp('Temperature test started.')

for k=1:N
    tic;

    % Computes analog time
    t(k) = (k-1)*Ts;

    % Reads the sensor temperatures
    y(1,k) = T1C();

    % Compute incremental variables
    Dy(:,k) = y(:,k) - y_ss;

    % ── Kalman filter CORRECTION step ────────────────────────────────────
    % xd_est(:,k) currently holds the PREDICTION made last iteration
    % (for k = 1 it holds the initialization). Correct it with Dy(k).
    xd_est(:,k) = xd_est(:,k) + L*(Dy(:,k) - Cd*xd_est(:,k));

    Dx_hat = xd_est(1:n,k);          % estimated incremental state
    d_hat  = xd_est(end,k);          % estimated input disturbance

    if USE_MPC
        % ── Feedforward: steady-state solve using the estimated d_hat ────
        % Clamp the reference to a reachable target so the MPC does not
        % chase the unreachable 60 degC leg against the 55 degC cap.
        %Dr_track = min(Dr(k), (y_max - margin) - y_ss);
        Dr_track = Dr(k);

        % Solve  (I-A)Dx_bar = B(Du_bar + d_hat),  C Dx_bar = Dr_track
        sol    = Mss \ [B*d_hat; Dr_track];
        Dx_bar = sol(1:n);
        Du_bar = sol(end);

        % Shifted control limits  du = Du - Du_bar,  Du in [-u_ss, 100-u_ss]
        lb = (-u_ss       - Du_bar) * ones(H,1);
        ub = ( 100 - u_ss - Du_bar) * ones(H,1);

        % Shifted output cap   y_hat(i) <= y_max  ->  dy_hat(i) <= y_max_inc
        y_max_inc = (y_max - y_ss) - Dr_track;

        % ── MPC: solve in shifted coordinates using the ESTIMATED state ──
        dx_k = Dx_hat - Dx_bar;

        if mode == 0
            [du_k, exitflag(k)] = mpc_solve(dx_k, H, R, A, B, C, ...
                                            lb, ub, y_max_inc, const_type);
        else
            [du_k, exitflag(k)] = mpc_solve_sparse_regularized(dx_k, H, R, ...
                                       A, B, C, lb, ub, y_max_inc, const_type);
        end

        Du(:,k) = du_k + Du_bar;     % incremental control
    else
        % Open-loop test of the Kalman filter: keep the equilibrium input.
        Du(:,k) = 0;
    end

    % Computes the absolute control variable to apply
    u(:,k) = Du(:,k) + u_ss;

    % Saturate to the physical heater range before applying (safety).
    u(:,k) = min(max(u(:,k),0),100);

    % ── Kalman filter PREDICTION step ────────────────────────────────────
    % Propagate to the next sample using the control just applied.
    xd_est(:,k+1) = Ad*xd_est(:,k) + Bd*Du(:,k);

    % Applies the control variable to the plant
    h1(u(1,k));

    if rt_plot
        % Plots results
        clf
        subplot(2,1,1), hold on, grid on
        plot(t(1:k),y(1,1:k),'.','MarkerSize',10)
        stairs(t,r,'g--')
        yline(y_max,'r--')
        xlabel('Time [s]')
        ylabel('y [°C]')
        subplot(2,1,2), hold on, grid on
        stairs(t(1:k),u(1,1:k),'LineWidth',2)
        xlabel('Time [s]')
        ylabel('u [%]')
        ylim([0 100]);
        drawnow;
    else
        fprintf('t = %d, y1 = %.1f C, u1 = %.1f\n',t(k),y(1,k),u(1,k)) %#ok<UNRCH>
    end

    % Check if computation time did not exceed sampling time
    if toc > Ts
        warning('Computation time exceeded sampling time by %.2f s at sample %d.',toc-Ts,k)
    end
    % Waits for the begining of the new sampling interval
    pause(max(0,Ts-toc));
end

% Turns off both heaters at the end of the experiment
h1(0);
h2(0);

% Signals the end of the experiment by shutting off the LED
led(0)

disp('Temperature test complete.')

%% Final plot ============================================================
% Reconstruct the estimated output for plotting / discussion.
y_hat = y_ss + C*xd_est(1:n,1:N);
d_hat_log = xd_est(end,1:N);

figure
subplot(3,1,1), hold on, grid on
plot(t(1:k),y(1,1:k),'.','MarkerSize',10)
plot(t(1:k),y_hat(1:k),'-','LineWidth',1.2)
stairs(t,r,'g--')
yline(y_max,'r--')
xlabel('Time [s]'), ylabel('y [°C]')
legend('measured y','estimated $\hat{y}$','reference r','$y_{max}$', ...
       'Interpreter','latex','Location','best')
title('P5 — Closed-loop experiment on the real plant')

subplot(3,1,2), hold on, grid on
stairs(t(1:k),u(1,1:k),'LineWidth',2)
yline(0,'r--'), yline(100,'r--'), yline(u_ss,'k--')
xlabel('Time [s]'), ylabel('u [%]')
ylim([0 100]);

subplot(3,1,3), hold on, grid on
plot(t(1:k),d_hat_log(1:k),'LineWidth',1.5)
xlabel('Time [s]'), ylabel('$\hat{d}$ [%]','Interpreter','latex')
legend('estimated $\hat{d}$','Location','best')

%--------------------------------------------------------------------------

% Save figure and experiment data to file
exportgraphics(gcf,['closedloop_plot_',timestr,'.png'],'Resolution',300)
save('closedloop_data_1.mat','y','u','t','r','y_hat','d_hat_log','exitflag');

%--------------------------------------------------------------------------
% End of File