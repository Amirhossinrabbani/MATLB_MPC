clc; clear; close all;

%% ==========================================1
s = tf('s');
G = (s + 4) / (s^2 + 1.2*s + 4);
Ts = 0.05;
G_disc = c2d(G, Ts, 'zoh');
[Ad, Bd, Cd, Dd] = ssdata(G_disc);
nx = size(Ad, 1);

%% =========================================
Np = 80;
Nu = 5;
lambda = 0.1;
umax = 0.5;
umin = -0.5;
OS = 0.05;

A_a = [Ad,zeros(nx,1);Cd*Ad,1];
B_a = [Bd; Cd*Bd];
C_a = [zeros(1,nx), 1];
nxa = size(A_a, 1);

F = zeros(Np, nxa);
for i = 1:Np
    F(i, :) = C_a * A_a^i;
end
Phi = zeros(Np, Nu);
for i = 1:Np
    for j = 1:min(i, Nu)
        Phi(i, j) = C_a * A_a^(i-j) * B_a;
    end
end

%% ==========================================================
t1 = 5; t2 = 10; t3 = 15; t_final = 20;
time = 0:Ts:t_final;
N = length(time);
r = zeros(N,1);
for i = 1:N
    t_i = time(i);
    if t_i < t1
        r(i) = 0;
    elseif t_i < t2
        r(i) = 1;
    elseif t_i < t3
        r(i) = -1;
    else
        r(i) = 0;
    end
end

%%=====================================================
xm = zeros(nx, 1);
xm_old = zeros(nx, 1);
u_prev = 0;
y = Cd * xm;

y_log = zeros(N, 1);
u_log = zeros(N, 1);
r_log = r;

r_fut = zeros(Np, 1);
A_os = zeros(Np, Nu);
b_os = zeros(Np, 1);

options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');

for k = 1:N
    if k < N
        len_fut = min(Np, N - k);
        r_fut(1:len_fut) = r(k+1 : k+len_fut);
        if len_fut < Np
            r_fut(len_fut+1:end) = r_fut(len_fut);
        end
    else
        r_fut(:) = r(N);
    end
    
    delta_xm = xm - xm_old;
    xa = [delta_xm; y];
    
    H_qp = 2 * (Phi' * Phi + lambda * eye(Nu));
    g_qp = 2 * Phi' * (F * xa - r_fut);
    
    T = tril(ones(Nu, Nu));
    A_u = [ T; -T];
    b_u = [ umax * ones(Nu, 1) - u_prev;
            -umin * ones(Nu, 1) + u_prev];
    
    row = 0;
    for i = 1:Np
        if r_fut(i) > 0
            row = row + 1;
            A_os(row, :) = Phi(i, :);
            b_os(row)    = (1+OS)*r_fut(i) - F(i,:)*xa;
        elseif r_fut(i) < 0
            row = row + 1;
            A_os(row, :) = -Phi(i, :);
            b_os(row)    = -(1+OS)*r_fut(i) + F(i,:)*xa;
        end
    end
    A_os_active = A_os(1:row, :);
    b_os_active = b_os(1:row);
    
    A_qp = [A_u; A_os_active];
    b_qp = [b_u; b_os_active];
    
    dU = quadprog(H_qp, g_qp, A_qp, b_qp, [], [], [], [], [], options);
    if isempty(dU)
        dU = zeros(Nu, 1);
    end
    
    du = dU(1);
    u = u_prev + du;
    u = max(min(u, umax), umin);
    
    u_log(k) = u;
    y_log(k) = y;
    
    xm_next = Ad * xm + Bd * u;
    y_next = Cd * xm_next;
    
    % ======================================================
    if time(k) >= t1 && time(k) < t2
        d = 0.2; 
    else
        d = 0;
    end
    y_measured = y_next + d;
    % ==================================================
    
    xm_old = xm;
    xm = xm_next;
    y = y_measured; 
    u_prev = u;
end

%%==========================================================================
figure;
subplot(2,1,1);
plot(time, r_log, 'k--', 'LineWidth', 1.5); hold on;
stairs(time, y_log, 'b', 'LineWidth', 1.5);
ylabel('خروجی y و مرجع r');
legend('مرجع r', 'خروجی y', 'Location', 'best');
grid on; xlim([0, t_final]);
title('GPC مقید با اغتشاش (بازه ۵-۱۰ ثانیه)');

subplot(2,1,2);
stairs(time, u_log, 'r', 'LineWidth', 1.5);
yline(umax, 'g--'); yline(umin, 'g--');
ylabel('ورودی u');
xlabel('زمان (ثانیه)');
legend('u', 'umax', 'umin', 'Location', 'best');
grid on; xlim([0, t_final]);