clear; clc; close all;
Ts = 0.05;
s = tf('s');

Gn = (s + 4) / (s^2 + 1.2*s + 4);
[Ad_n, Bd_n, Cd_n, Dd_n] = ssdata(c2d(Gn, Ts, 'zoh'));
nx = size(Ad_n,1);

poles = pole(Gn);

p1 = poles(1) * 1.2;
p2 = poles(2) * 1.2;

den_r = [1, -(p1+p2), p1*p2];
num_r = [1, 4];
Gr = tf(num_r, den_r);

dc_gain_r = dcgain(Gr);
Gr = Gr / dc_gain_r;

Gr_disc = c2d(Gr, Ts, 'zoh');
[Ad_r, Bd_r, Cd_r, Dd_r] = ssdata(Gr_disc);

% ================================================================
poles_r = pole(Gr);

t1=5; t2=10; t3=15; t_final=20;
time = 0:Ts:t_final;
N = length(time);
r = zeros(N,1);
for i=1:N
    t_i = time(i);
    if t_i < t1
        r(i)=0;
    elseif t_i < t2
        r(i)=1;
    elseif t_i < t3
        r(i)=-1;
    else
        r(i)=0;
    end
end

%% ============================== PFC ==============================
Hp = 10;
N_L = 6;
a_lag = 0.5;
lambda_u = 6;
tau_ref = 1.0;

L = laguerre_matrix(N_L, Hp, a_lag);
[Phi, Gamma] = pred_matrices(Ad_n, Bd_n, Cd_n, Hp);
Psi = Gamma * L;
lambda_ref = exp(-Ts / tau_ref);

x_real = zeros(nx, N);
u_pfc = zeros(N,1);
y_pfc = zeros(N,1);
x_real(:,1) = zeros(nx,1);
y_pfc(1) = Cd_r * x_real(:,1);

umax_pfc = 2;  umin_pfc = -2;
overshoot_percent = 0.05;

max_constraints = 3*Hp;
A_cons = zeros(max_constraints, N_L);
b_cons = zeros(max_constraints,1);
options_qp = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');

for k = 1:N-1
    r_k = r(k);
    if abs(r_k) > 0.5
        y_max_limit = r_k + overshoot_percent*abs(r_k);
        y_min_limit = r_k - overshoot_percent*abs(r_k);
    else
        y_max_limit = overshoot_percent;
        y_min_limit = -overshoot_percent;
    end
    w = r_k * ones(Hp,1);
    
    constraint_count = 0;
    for i=1:Hp
        constraint_count = constraint_count + 1;
        A_cons(constraint_count,:) = Psi(i,:);
        b_cons(constraint_count) = y_max_limit - Phi(i,:)*x_real(:,k);
        
        constraint_count = constraint_count + 1;
        A_cons(constraint_count,:) = -Psi(i,:);
        b_cons(constraint_count) = -y_min_limit + Phi(i,:)*x_real(:,k);
    end
    for i=1:Hp
        constraint_count = constraint_count + 1;
        A_cons(constraint_count,:) = L(i,:);
        b_cons(constraint_count) = umax_pfc;
        
        constraint_count = constraint_count + 1;
        A_cons(constraint_count,:) = -L(i,:);
        b_cons(constraint_count) = -umin_pfc;
    end
    A_cons = A_cons(1:constraint_count,:);
    b_cons = b_cons(1:constraint_count);
    
    H = Psi'*Psi + lambda_u*(L'*L);
    f = -(w - Phi*x_real(:,k))' * Psi;
    
    eta_opt = quadprog(H, f, A_cons, b_cons, [], [], [], [], [], options_qp);
    if isempty(eta_opt)
        if k>1, u_pfc(k)=u_pfc(k-1); else u_pfc(k)=0; end
    else
        u_pfc(k) = L(1,:) * eta_opt;
    end
    u_pfc(k) = max(umin_pfc, min(umax_pfc, u_pfc(k)));
    
    x_real(:,k+1) = Ad_r * x_real(:,k) + Bd_r * u_pfc(k);
    y_pfc(k+1) = Cd_r * x_real(:,k+1);
end

%% ============================== GPC ==============================
Np = 80;
Nu = 5;
lambda = 0.1;
umax_gpc = 0.5;
umin_gpc = -0.5;
OS = 0.05;

A_a = [Ad_n, zeros(nx,1); Cd_n*Ad_n, 1];
B_a = [Bd_n; Cd_n*Bd_n];
C_a = [zeros(1,nx), 1];
nxa = size(A_a,1);

F = zeros(Np, nxa);
for i=1:Np
    F(i,:) = C_a * A_a^i;
end
Phi_gpc = zeros(Np, Nu);
for i=1:Np
    for j=1:min(i,Nu)
        Phi_gpc(i,j) = C_a * A_a^(i-j) * B_a;
    end
end

xm_real = zeros(nx,1);
xm_old = zeros(nx,1);
u_prev = 0;
y_gpc = Cd_r * xm_real;
y_gpc_log = zeros(N,1);
u_gpc_log = zeros(N,1);
y_gpc_log(1) = y_gpc;

r_fut = zeros(Np,1);
A_os = zeros(Np, Nu);
b_os = zeros(Np,1);

options_qp2 = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');

for k = 1:N
    if k < N
        len_fut = min(Np, N-k);
        r_fut(1:len_fut) = r(k+1 : k+len_fut);
        if len_fut < Np
            r_fut(len_fut+1:end) = r_fut(len_fut);
        end
    else
        r_fut(:) = r(N);
    end
    
    delta_xm = xm_real - xm_old;
    xa = [delta_xm; y_gpc];
    
    H_qp = 2*(Phi_gpc'*Phi_gpc + lambda*eye(Nu));
    g_qp = 2*Phi_gpc'*(F*xa - r_fut);
    
    T = tril(ones(Nu,Nu));
    A_u = [T; -T];
    b_u = [umax_gpc*ones(Nu,1) - u_prev; -umin_gpc*ones(Nu,1) + u_prev];
    
    row = 0;
    for i=1:Np
        if r_fut(i) > 0
            row = row+1;
            A_os(row,:) = Phi_gpc(i,:);
            b_os(row) = (1+OS)*r_fut(i) - F(i,:)*xa;
        elseif r_fut(i) < 0
            row = row+1;
            A_os(row,:) = -Phi_gpc(i,:);
            b_os(row) = -(1+OS)*r_fut(i) + F(i,:)*xa;
        end
    end
    A_os_active = A_os(1:row,:);
    b_os_active = b_os(1:row);
    
    A_qp = [A_u; A_os_active];
    b_qp = [b_u; b_os_active];
    
    dU = quadprog(H_qp, g_qp, A_qp, b_qp, [], [], [], [], [], options_qp2);
    if isempty(dU)
        dU = zeros(Nu,1);
    end
    du = dU(1);
    u = u_prev + du;
    u = max(umin_gpc, min(umax_gpc, u));
    
    u_gpc_log(k) = u;
    y_gpc_log(k) = y_gpc;
    
    xm_next = Ad_r * xm_real + Bd_r * u;
    y_next = Cd_r * xm_next;
    
    xm_old = xm_real;
    xm_real = xm_next;
    y_gpc = y_next;
    u_prev = u;
end

%%================================================================== IAE 
IAE_pfc = sum(abs(y_pfc - r)) * Ts;
IAE_gpc = sum(abs(y_gpc_log - r)) * Ts;

overshoot_pfc = [0,0,0];
overshoot_gpc = [0,0,0];
idx1 = find(r==1);
if ~isempty(idx1)
    overshoot_pfc(1) = (max(y_pfc(idx1)) - 1) / 1 * 100;
    overshoot_gpc(1) = (max(y_gpc_log(idx1)) - 1) / 1 * 100;
end
idx2 = find(r==-1);
if ~isempty(idx2)
    overshoot_pfc(2) = (min(y_pfc(idx2)) - (-1)) / 1 * 100;
    overshoot_gpc(2) = (min(y_gpc_log(idx2)) - (-1)) / 1 * 100;
end
idx3 = find(time>t3 & time<=t3+2);
if ~isempty(idx3)
    overshoot_pfc(3) = max(y_pfc(idx3)) * 100;
    overshoot_gpc(3) = max(y_gpc_log(idx3)) * 100;
end

%% ============================================================================================

figure('Position', [100 100 1400 900]);

subplot(2,3,1);
plot(time, r, 'k--', 'LineWidth', 1.5); hold on;
plot(time, y_pfc, 'b-', 'LineWidth', 1.2);
plot(time, y_gpc_log, 'r-', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Output');
title('مقایسه خروجی PFC و GPC روی سیستم واقعی');
legend('r', 'PFC', 'GPC', 'Location','best');
grid on; ylim([-1.5 1.8]);


subplot(2,3,[2;3]);
stairs(time, u_pfc, 'b-', 'LineWidth', 1); hold on;
stairs(time, u_gpc_log, 'r-', 'LineWidth', 1);
yline(umax_pfc,'g--','u_{max}=2'); yline(umin_pfc,'g--','u_{min}=-2');
yline(umax_gpc,'m--','u_{max}=0.5'); yline(umin_gpc,'m--','u_{min}=-0.5');
xlabel('Time (s)'); ylabel('Control');
title('سیگنال کنترل');
legend('PFC', 'GPC', 'Location','best');
grid on;

subplot(2,3,4);
plot(time, abs(y_pfc - r), 'b-', 'LineWidth', 1); hold on;
plot(time, abs(y_gpc_log - r), 'r-', 'LineWidth', 1);
xlabel('Time (s)'); ylabel('|y-r|');
title('خطای مطلق ردیابی');
legend('PFC', 'GPC');
grid on;

subplot(2,3,5);
axis off;
text(0.1, 0.9, sprintf('======= شاخص‌های عملکرد =======\n'), 'FontSize', 10);
text(0.1, 0.8, sprintf('IAE (PFC): %.4f\n', IAE_pfc));
text(0.1, 0.7, sprintf('IAE (GPC): %.4f\n', IAE_gpc));
text(0.1, 0.6, sprintf('os  0→1: PFC=%.2f%%, GPC=%.2f%%\n', overshoot_pfc(1), overshoot_gpc(1)));
text(0.1, 0.5, sprintf('os 1→-1: PFC=%.2f%%, GPC=%.2f%%\n', overshoot_pfc(2), overshoot_gpc(2)));
text(0.1, 0.4, sprintf('os -1→0: PFC=%.2f%%, GPC=%.2f%%\n', overshoot_pfc(3), overshoot_gpc(3)));

subplot(2,3,6);
bode(Gn, Gr, {0.1, 100});
title('پاسخ فرکانسی مدل نامی و واقعی');
legend('مدل نامی', 'سیستم واقعی (قطب‌ها 20% جابجا شده)');
grid on;

sgtitle('مقایسه مقاومت PFC و GPC در برابر عدم قطعیت 20% جابجایی قطب‌ها');

%% ===================================================================================
function L = laguerre_matrix(N, Np, a)
    L = zeros(Np, N);
    L(1,1) = sqrt(1 - a^2);
    for j = 2:N
        L(1,j) = -a * L(1, j-1);
    end
    for i = 2:Np
        L(i,1) = a * L(i-1,1);
        for j = 2:N
            L(i,j) = a * L(i-1,j) + L(i-1,j-1) - a * L(i,j-1);
        end
    end
end

function [Phi, Gamma] = pred_matrices(A, B, C, Hp)
    nx = size(A,1);
    Phi = zeros(Hp, nx);
    Gamma = zeros(Hp, Hp);
    for i = 1:Hp
        Phi(i,:) = C * A^i;
        for j = 1:i
            Gamma(i,j) = C * A^(i-j) * B;
        end
    end
end