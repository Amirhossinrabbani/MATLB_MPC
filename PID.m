clc; clear; close all;

num = 1;
den = [1, 0.7, 5];
G_continuous = tf(num, den);

Ts = 0.05;
G_discrete = c2d(G_continuous, Ts, 'zoh');
%%============================================================================

t1 = 5;
t2 = 10;
t3 = 15;
t_final = 20;

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

figure('Name', 'Reference Signal', 'Position', [100, 100, 800, 400]);
plot(time, r, 'b-', 'LineWidth', 2);
grid on;
xlabel('time');
ylabel('y');
title('refrence trajectory r(t)');
xlim([0, t_final]);

%%=============================================================================

Kp = 15; 
Ki = 10;
Kd = 2;
C_pid_continuous = pid(Kp, Ki, Kd);

C_pid_discrete = c2d(C_pid_continuous, Ts, 'tustin');

system_pid = feedback(C_pid_discrete * G_discrete, 1);

y_pid = lsim(system_pid, r, time);

[A_G, B_G, C_G, D_G] = ssdata(G_discrete);
[A_C, B_C, C_C, D_C] = ssdata(C_pid_discrete);


n_xg = size(A_G, 1);
n_xc = size(A_C, 1); 

A_cl = [A_G - B_G*D_C*C_G, -B_G*C_C;
        B_C*C_G, A_C];
    
B_cl = [B_G*D_C; 
        B_C];

x = zeros(n_xg + n_xc, 1); 
y_pid2 = zeros(N, 1);
u_pid = zeros(N, 1);

for k = 1:N
    error = r(k) - y_pid2(k);
 
    if k == 1
        u_pid(k) = Kp * error; 
    else
        if k == 2
            sum_error = error * Ts;
            u_pid(k) = Kp * error + Ki * sum_error + Kd * (error - 0) / Ts;
            e_prev = error;
        else
            sum_error = sum_error + error * Ts;
            u_pid(k) = Kp * error + Ki * sum_error + Kd * (error - e_prev) / Ts;
            e_prev = error;
        end
    end
    
    u_pid(k) = max(-5, min(5.5, u_pid(k)));
    
    x_G = x(1:n_xg);
    y_new = C_G * x_G + D_G * u_pid(k);
    y_pid2(k) = y_new;
    
    x_G_new = A_G * x_G + B_G * u_pid(k);
    x(1:n_xg) = x_G_new;
end

%%========================================================================================
e_pid = r' - y_pid;
u_pid_simple = zeros(N, 1);
sum_e = 0;
e_prev = 0;

for k = 1:N
    sum_e = sum_e + e_pid(k) * Ts;
    u_pid_simple(k) = Kp * e_pid(k) + Ki * sum_e + Kd * (e_pid(k) - e_prev) / Ts;
    e_prev = e_pid(k);
    
    u_pid_simple(k) = max(-5, min(5.5, u_pid_simple(k)));
end

%%==========================================================================================

figure('Position', [100, 100, 1200, 800]);

subplot(2,1,1);
plot(time, r, 'b-', 'LineWidth', 2, 'DisplayName', 'ورودی مرجع (r)');
hold on;
plot(time, y_pid, 'r--', 'LineWidth', 2, 'DisplayName', 'خروجی سیستم (y)');
grid on;
xlabel('time ');
ylabel(' y ');
title('step response with PID');
legend('Location', 'best');
xlim([0, 20]);
ylim([-1.5, 1.5]);

subplot(2,1,2);
plot(time, u_pid_simple, 'g-', 'LineWidth', 2, 'DisplayName', ' control signal u(t)');
hold on;
plot([0, t_final], [5.5, 5.5], 'r--', 'LineWidth', 1.5, 'DisplayName', 'range u=5.5');
plot([0, t_final], [-5, -5], 'r--', 'LineWidth', 1.5, 'DisplayName', 'renge u=-5');
grid on;
xlabel('time');
ylabel('control signal');
title('PID signal');
legend('Location', 'best');
xlim([0, 20]);

%%-==========================================================================================

idx_pulse1_start = find(time >= t1, 1);
idx_pulse1_end = find(time >= t2, 1);
y_step1 = y_pid(idx_pulse1_start:idx_pulse1_end);
max_y1 = max(y_step1);
overshoot1 = max(0, (max_y1 - 1) / 1 * 100);

idx_pulse2_start = find(time >= t2, 1);
idx_pulse2_end = find(time >= t3, 1);
y_step2 = y_pid(idx_pulse2_start:idx_pulse2_end);
min_y2 = min(y_step2);

undershoot2 = max(0, (abs(min_y2 + 1)) / 1 * 100);

fprintf('\n======= Result PID =======\n');
fprintf('overshot step in t1: %.2f %%\n', overshoot1);
fprintf('unbershot step in t2 : %.2f %%\n', undershoot2);
fprintf('control signal range: min = %.2f, max = %.2f\n', min(u_pid_simple), max(u_pid_simple));


