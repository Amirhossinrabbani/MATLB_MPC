clc; clear; close all;

% =========================================================================
num = 1;
den = [1, 0.7, 5];
G_continuous = tf(num, den);
Ts = 0.05;
G_discrete = c2d(G_continuous, Ts, 'zoh');

% =========================================================================

N1 = 1;  
N2 = 10; 
Nu = 3;
lambda = 0.1;

[A, B, C, D] = ssdata(G_discrete);
n = size(A, 1);

[num_b, den_b] = tfdata(G_discrete, 'v');
B_z = num_b;
A_z = den_b;

A_tilde = conv(A_z, [1, -1]);
A_tilde = A_tilde(2:end);

g = step(G_discrete, (0:N2-1)*Ts);
g = g(2:end); 

G = zeros(N2 - N1 + 1, Nu);
for i = 1:(N2 - N1 + 1)
    for j = 1:Nu
        if i-j+1 > 0 && i-j+1 <= length(g)
            G(i, j) = g(i-j+1);
        end
    end
end

% =========================================================================

t1 = 5; t2 = 10; t3 = 15; t_final = 20;
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

y_gpc = zeros(N, 1);
u_gpc = zeros(N, 1);
x_state = zeros(n, 1); 
u_prev = 0;
y_prev = 0;

for k = 1:N
    r_future = zeros(N2 - N1 + 1, 1);
    for j = N1:N2
        idx = min(k + j, N);
        r_future(j - N1 + 1) = r(idx);
    end
    
    y_free = 0;
    for j = 1:length(A_tilde)-1
        if k-j > 0
            y_free = y_free - A_tilde(j+1) * y_gpc(k-j);
        end
    end
    for j = 1:length(B_z)
        if k-j > 0
            y_free = y_free + B_z(j) * u_gpc(k-j);
        end
    end

    R = lambda * eye(Nu);
    u_opt = (G' * G + R) \ (G' * (r_future - y_free));
    
    u_gpc(k) = u_opt(1);
    
    u_gpc(k) = max(-5, min(5.5, u_gpc(k)));
    
    x_state = A * x_state + B * u_gpc(k);
    y_gpc(k) = C * x_state + D * u_gpc(k);
end


% =========================================================================

figure('Position', [100, 100, 1400, 900]);

subplot(2,2,[1,2]);
plot(time, r, 'b-', 'LineWidth', 2, 'DisplayName', 'refrence(r)');
hold on;
plot(time, y_gpc, 'r-', 'LineWidth', 2, 'DisplayName', 'GPC');
grid on;
xlabel('time');
ylabel('y(t)');
legend('Location', 'best');
xlim([0, t_final]);
ylim([-1.5, 1.5]);

subplot(2,2,3);
plot(time, u_gpc, 'r-', 'LineWidth', 2, 'DisplayName', 'u_{GPC}(t)');
hold on;
plot([0, t_final], [5.5, 5.5], 'k--', 'LineWidth', 1.5, 'DisplayName', 'uper band');
plot([0, t_final], [-5, -5], 'k--', 'LineWidth', 1.5, 'DisplayName', 'lower bund');
grid on;
xlabel('time');
ylabel('control signal');
title(' control signal GPC');
legend('Location', 'best');
xlim([0, t_final]);

subplot(2,2,4);
error_gpc = r - y_gpc';
plot(time, error_gpc, 'r-', 'LineWidth', 2, 'DisplayName', 'خطای GPC');
hold on;
plot([0, t_final], [0, 0], 'k-', 'LineWidth', 1);
grid on;
xlabel('time');
ylabel(' renge error e(t)');
legend('Location', 'best');
xlim([0, t_final]);

% =========================================================================
IAE_gpc = sum(abs(error_gpc)) * Ts;
ISE_gpc = sum(error_gpc.^2) * Ts;
ITAE_gpc = sum(time' .* abs(error_gpc)) * Ts;

idx1 = find(time >= t1, 1);
idx2 = find(time >= t2, 1);
y_step_gpc = y_gpc(idx1:idx2);
overshoot_gpc = max(0, (max(y_step_gpc) - 1) / 1 * 100);

idx3 = find(time >= t3, 1);
y_step2_gpc = y_gpc(idx2:idx3);
undershoot_gpc = max(0, abs(min(y_step2_gpc) + 1) / 1 * 100);
%%========================================================================

fprintf('\n========== GPC ==========\n');
fprintf('overshot step in t1: %.2f %%\n', overshoot_gpc);
fprintf('overshot step in t2: %.2f %%\n', undershoot_gpc);
fprintf('IAE: %.4f\n', IAE_gpc);
fprintf('ISE: %.4f\n', ISE_gpc);
fprintf('ITAE: %.4f\n', ITAE_gpc(end));


