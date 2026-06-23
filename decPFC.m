clear; clc; close all;
s = tf('s');
G = (s + 4) / (s^2 + 1.2*s + 4);
Ts = 0.05;
G_disc = c2d(G, Ts, 'zoh');
[Ad, Bd, Cd, Dd] = ssdata(G_disc);
nx = size(Ad,1);

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

Hp = 10;
N_L = 6;
a_lag = 0.5;
lambda_u = 6;
tau_ref = 1.0;
L = laguerre_matrix(N_L, Hp, a_lag);
[Phi, Gamma] = pred_matrices(Ad, Bd, Cd, Hp);
Psi = Gamma * L;

x = zeros(nx, N);
u = zeros(N,1);
y = zeros(N,1);
x(:,1) = zeros(nx,1);
y(1) = Cd * x(:,1);

umax = 2;  umin = -2;
overshoot_percent = 0.05;
max_constraints = 3 * Hp;
A_cons = zeros(max_constraints, N_L);
b_cons = zeros(max_constraints, 1);

for k = 1:N-1
    r_k = r(k);
    if abs(r_k) > 0.5
        y_max_limit = r_k + overshoot_percent * abs(r_k);
        y_min_limit = r_k - overshoot_percent * abs(r_k);
    else
        y_max_limit = overshoot_percent;
        y_min_limit = -overshoot_percent;
    end
    
    w = r_k * ones(Hp,1);
    
    A_cons(:) = 0;
    b_cons(:) = 0;
    constraint_count = 0;
    
    for i = 1:Hp
        constraint_count = constraint_count + 1;
        A_cons(constraint_count, :) = Psi(i,:);
        b_cons(constraint_count) = y_max_limit - Phi(i,:)*x(:,k);
        
        constraint_count = constraint_count + 1;
        A_cons(constraint_count, :) = -Psi(i,:);
        b_cons(constraint_count) = -y_min_limit + Phi(i,:)*x(:,k);
    end
    
    for i = 1:Hp
        constraint_count = constraint_count + 1;
        A_cons(constraint_count, :) = L(i,:);
        b_cons(constraint_count) = umax;
        
        constraint_count = constraint_count + 1;
        A_cons(constraint_count, :) = -L(i,:);
        b_cons(constraint_count) = -umin;
    end
    
    A_cons = A_cons(1:constraint_count, :);
    b_cons = b_cons(1:constraint_count);
    
    H = Psi' * Psi + lambda_u * (L' * L);
    f = - (w - Phi*x(:,k))' * Psi;
    
    options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
    eta_opt = quadprog(H, f, A_cons, b_cons, [], [], [], [], [], options);
    
    if isempty(eta_opt)
        if k > 1
            u(k) = u(k-1);
        else
            u(k) = 0;
        end
    else
        u(k) = L(1,:) * eta_opt;
    end
    
    u(k) = max(umin, min(umax, u(k)));
    
    d_input = 0;
    if time(k+1) >= t1 && time(k+1) < t2 
        d_input = 0.2; 
    end
    
    % ============================
    x(:,k+1) = Ad * x(:,k) + Bd * (u(k) + d_input);
    
    y(k+1) = Cd * x(:,k+1);
    % ===============================================================
end

figure('Position', [100 100 1200 800]);
subplot(2,1,1);
plot(time, r, 'k--', 'LineWidth', 2); hold on;
plot(time, y, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Output y');
title('PFC با قیود و اغتشاش در بازه ۵-۱۰ ثانیه');
legend('Reference', 'Output');
grid on; ylim([-1.5 1.5]);

subplot(2,1,2);
stairs(time, u, 'r-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Control u');
title('سیگنال کنترل');
yline(2,'g--', 'u_{max}=2'); yline(-2,'g--', 'u_{min}=-2');
legend('u(k)', 'Limits');
grid on; ylim([-2.5 2.5]);

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