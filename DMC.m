clc; clear; close all;

%%==============================================================
num = 1;
den = [1, 0.7, 5];
G_continuous = tf(num, den);
Ts = 0.05;
G_discrete = c2d(G_continuous, Ts, 'zoh');

%% =============================================================
t1 = 5;   t2 = 10;  t3 = 15; t_final = 20;
time = 0:Ts:t_final;
N = length(time);

r = zeros(1, N);
for k = 1:N
    t = time(k);
    if t >= t1 && t < t2
        r(k) = 1;
    elseif t >= t2 && t < t3
        r(k) = -1;
    elseif t >= t3
        r(k) = 0;
    end
end

%% =============================  DMC  ===============

N_model = 120;
t_step = (0:N_model-1)' * Ts;
[y_step, ~] = step(G_discrete, t_step(end));
g = y_step(1:N_model); 
g_inf = g(end); 

Hp = 20;
Hu = 5; 
lambda = 0.05;

%%====================================================== 
D = zeros(Hp, Hu);
for i = 1:Hp
    for j = 1:min(i, Hu)
        D(i, j) = g(i - j + 1);
    end
end

%%===================================================== 
I_lambda = lambda * eye(Hu);
K_dmc_full = (D' * D + I_lambda) \ D';
K_dmc = K_dmc_full(1, :); 

%%====================================================== 
y = zeros(N, 1); 
u = zeros(N, 1);
du = zeros(N, 1); 

[A_d, B_d, C_d, D_ss] = ssdata(G_discrete);
x_state = zeros(size(A_d, 1), 1);

%% =====================================================
for k = 1:N
    
    Y0 = zeros(Hp, 1);

    y_current = y(k);
    
    for i = 1:Hp 
        
        if k > 1
            Y0(i) = Y0(i) + g_inf * u(k-1);
        end
        
        for m = 1:min(N_model-i, k-1)
            idx = k - m;  
            if idx >= 1
                Y0(i) = Y0(i) + g(m+i) * du(idx);
            end
        end

        if i == 1
            model_error = y_current - Y0(1);
        end
        Y0(i) = Y0(i) + model_error;
    end
    
    %% ----- (W) -----
    W = zeros(Hp, 1);
    for i = 1:Hp
        idx_future = min(k + i - 1, N);
        W(i) = r(idx_future);
    end
    
    %% ----- error-----
    error_pred = W - Y0;
    delta_U = K_dmc * error_pred;
    delta_u = delta_U(1);
    
    %% ----- range -----
    delta_u_max = 0.8;
    delta_u = max(-delta_u_max, min(delta_u_max, delta_u));
    
    %% ----- new input -----
    if k == 1
        u_new = delta_u;
    else
        u_new = u(k-1) + delta_u;
    end
    
    
    u_new = max(-5, min(5.5, u_new));
    
    % ذخیره مقادیر
    u(k) = u_new;
    if k == 1
        du(k) = u_new;
    else
        du(k) = u(k) - u(k-1);
    end
    
    %% -----  output -----
    y(k+1:min(k+1,N)) = 0;
    x_state = A_d * x_state + B_d * u(k);
    if k < N
        y(k+1) = C_d * x_state + D_ss * u(k);
    end
end

%% =============================================================

figure('Position', [100, 100, 1200, 800]);

subplot(2,1,1);
stairs(time, r, 'b-', 'LineWidth', 2); hold on;
plot(time, y, 'r-', 'LineWidth', 2);
plot([t1 t1], [-1.5 1.5], 'k:', 'LineWidth', 1.5);
plot([t2 t2], [-1.5 1.5], 'k:', 'LineWidth', 1.5);
plot([t3 t3], [-1.5 1.5], 'k:', 'LineWidth', 1.5);
grid on;
xlabel('time'); ylabel('y');
title('Responce with DMC');
legend('refrence (r)', ' out put (y)', 'Location', 'best');
xlim([0, 20]); ylim([-1.5, 1.5]);


subplot(2,1,2);
stairs(time, u, 'g-', 'LineWidth', 2); hold on;
plot([0, t_final], [5.5, 5.5], 'r--', 'LineWidth', 1.5);
plot([0, t_final], [-5, -5], 'r--', 'LineWidth', 1.5);
plot([t1 t1], [-6 6], 'k:', 'LineWidth', 1.5);
plot([t2 t2], [-6 6], 'k:', 'LineWidth', 1.5);
plot([t3 t3], [-6 6], 'k:', 'LineWidth', 1.5);
grid on;
xlabel('time'); ylabel('signal control');
title('signal control');
legend('u(t)', 'uper band (5.5)', 'lower bnd (-5)', 'Location', 'best');
xlim([0, 20]); ylim([-6, 6]);

%%================================================================ 
idx1 = find(time >= t1, 1);
idx2 = find(time >= t2, 1);
idx3 = find(time >= t3, 1);

y_seg1 = y(idx1:min(idx2-1, N));
if ~isempty(y_seg1) && max(y_seg1) > 1
    overshoot1 = (max(y_seg1) - 1) * 100;
else
    overshoot1 = max(0, (max(y_seg1) - 1) * 100);
end

y_seg2 = y(idx2:min(idx3-1, N));
if ~isempty(y_seg2) && min(y_seg2) < -1
    overshoot2 = (-1 - min(y_seg2)) * 100;
else
    overshoot2 = 0;
end


fprintf('\n========== Result DMC ==========\n');
fprintf(' Hp=%d, Hu=%d, λ=%.3f\n', Hp, Hu, lambda);
fprintf('overshot step in t1:  %.2f %%\n', overshoot1);
fprintf('overshot step in t2: %.2f %%\n', overshoot2);
