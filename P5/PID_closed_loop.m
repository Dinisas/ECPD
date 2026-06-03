% Closed-loop experiment for data collection in TCLab -- PID version
%
% Replaces the MPC + Kalman filter of P5 with a discrete PI/PID controller.
% Same reference schedule and the same y_max safety constraint as P5.
%
% The reference is CLAMPED to (y_max - margin) before the controller sees
% it. This is essential for a PI controller: without the clamp, an
% unreachable setpoint (e.g. 65 degC against a 55 degC cap) would cause
% massive integrator wind-up, and the system would overshoot wildly once
% the reference dropped back to a reachable value.
%
% Recommended workflow:
%   1. Tune (Kp, Ki, Kd) in PID_simulation.m first.
%   2. Copy the tuned gains here.
%   3. Run.
%
% Functions called: tclab.
%
% Adapted from the P5 MPC + Kalman closed-loop script.
%__________________________________________________________________________

% Initialization
clear all
close all
clc

base = fileparts(mfilename('fullpath'));
addpath(fullfile(base,'..','TCLab_software_test'));
load(fullfile(base,'..','P4','P4_4','singleheater_model2D.mat'),'y_ss','u_ss','Ts');
tclab;

%% Experiment parameters ===============================================
r_levels = [50 40 65 45];           % absolute reference [degC]
seg_time = 400;                     % seconds held at each level
% Longer / harder schedule:
% r_levels = [50 40 65 45 30 80 50 65 35 20 50];
% seg_time = 600;

T = seg_time*numel(r_levels);       % experiment duration [s]
N = round(T/Ts);                    % number of samples to collect

%% Safety constraint ===================================================
y_max  = 55;            % hard safety cap [degC]
margin = 0.5;           % clamp reference this far BELOW the cap [degC]

%% PID tuning -- copy from the simulation =============================
Kp = 15.0;    % proportional gain         [%/degC]
Ki = 0.10;   % integral gain             [%/(degC*s)]
Kd = 0.0;    % derivative gain (0 = PI)  [%*s/degC]

% Anti-windup: 'clamp' (conditional integration) or 'backcalc'.
awup = 'clamp';
Kt   = sqrt(Kp*Ki);

%% Reference signal ====================================================
r = zeros(1,N);
for s = 1:numel(r_levels)
    idx = (1:N) > (s-1)*seg_time/Ts & (1:N) <= s*seg_time/Ts;
    r(idx) = r_levels(s);
end

%% Storage =============================================================
% Real-time plot flag. If true, plots in real time. Otherwise prints.
rt_plot = true;

if rt_plot
    figure
    drawnow;
end
t         = nan(1,N);
u         = zeros(1,N);
y         = zeros(1,N);
e_log     = nan(1,N);
I_log     = nan(1,N);
r_track_l = nan(1,N);
sat_log   = false(1,N);

% PID internal state
I_term = 0;
e_prev = 0;
y_prev = NaN;       % set after first measurement

% String with date for saving results
timestr = char(datetime('now','Format','yyMMdd_HHmmSS'));

% Signals the start of the experiment by lighting the LED
led(1)
disp('Temperature test started.')

for k = 1:N
    tic;

    % Analog time
    t(k) = (k-1)*Ts;

    % ── 1. SENSE ──────────────────────────────────────────────────────
    y(1,k) = T1C();

    if k == 1
        y_prev = y(1,k);    % initialise on first sample
    end

    % ── 2. REFERENCE (clamped with margin) ────────────────────────────
    r_track       = min(r(k), y_max - margin);
    r_track_l(k)  = r_track;

    % ── 3. PID ────────────────────────────────────────────────────────
    e = r_track - y(1,k);
    e_log(k) = e;

    % Proportional
    P_term = Kp * e;

    % Tentative integral
    I_tent = I_term + Ki * Ts * e;

    % Derivative ON MEASUREMENT (no setpoint kick)
    if k == 1
        D_term = 0;
    else
        D_term = -Kd * (y(1,k) - y_prev) / Ts;
    end

    % Tentative (unsaturated) command
    u_tent = u_ss + P_term + I_tent + D_term;

    % Saturate to [0, 100]
    u_k = min(max(u_tent, 0), 100);
    sat_log(k) = (u_k ~= u_tent);

    % Anti-windup
    switch awup
        case 'clamp'
            if (u_k == u_tent) || (sign(e) ~= sign(u_tent - u_k))
                I_term = I_tent;
            end
        case 'backcalc'
            I_term = I_tent + Kt * Ts * (u_k - u_tent);
        otherwise
            error('Unknown anti-windup mode: %s', awup);
    end

    I_log(k) = I_term;
    u(1,k)   = u_k;
    e_prev   = e;
    y_prev   = y(1,k);

    % ── 4. ACT: apply to the plant ─────────────────────────────────────
    h1(u(1,k));

    if rt_plot
        clf
        subplot(2,1,1), hold on, grid on
        plot(t(1:k),y(1,1:k),'.','MarkerSize',10)
        stairs(t,r,'g--')
        yline(y_max,'r--')
        xlabel('Time [s]')
        ylabel('y [°C]')
        title(sprintf('PID: K_p=%.2f  K_i=%.3f  K_d=%.2f', Kp, Ki, Kd))
        subplot(2,1,2), hold on, grid on
        stairs(t(1:k),u(1,1:k),'LineWidth',2)
        xlabel('Time [s]')
        ylabel('u [%]')
        ylim([0 100]);
        drawnow;
    else
        fprintf('t = %d, y1 = %.1f C, u1 = %.1f\n', t(k), y(1,k), u(1,k)) %#ok<UNRCH>
    end

    % Check sampling-time budget
    if toc > Ts
        warning('Computation time exceeded sampling time by %.2f s at sample %d.', toc-Ts, k)
    end
    pause(max(0,Ts-toc));
end

% Turn off heaters at the end
h1(0);
h2(0);
led(0)
disp('Temperature test complete.')

%% Final plot ============================================================
figure
subplot(3,1,1), hold on, grid on
plot(t(1:k),y(1,1:k),'.','MarkerSize',10)
stairs(t,r,'g--')
stairs(t,r_track_l,'b:','LineWidth',1.0)
yline(y_max,'r--')
xlabel('Time [s]'), ylabel('y [°C]')
legend('measured y','reference r','clamped r_{track}','y_{max}', ...
       'Location','best')
title(sprintf('PID closed-loop — K_p=%.2f, K_i=%.3f, K_d=%.2f', Kp, Ki, Kd))

subplot(3,1,2), hold on, grid on
stairs(t(1:k),u(1,1:k),'LineWidth',2)
yline(0,'r--'), yline(100,'r--'), yline(u_ss,'k--')
xlabel('Time [s]'), ylabel('u [%]')
ylim([0 100]);

subplot(3,1,3), hold on, grid on
plot(t(1:k),e_log(1:k),'LineWidth',1.2)
yline(0,'k--')
xlabel('Time [s]'), ylabel('error e [°C]')

%--------------------------------------------------------------------------

% Save figure and experiment data
exportgraphics(gcf, ['PID_closedloop_plot_',timestr,'.png'], 'Resolution', 300)
save(['PID_closedloop_data_',timestr,'.mat'], ...
     'y','u','t','r','r_track_l','e_log','I_log','sat_log','Kp','Ki','Kd');

%--------------------------------------------------------------------------
% End of File