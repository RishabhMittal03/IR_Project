% IBVS Simulation IR Project
% i_curr: [i_curr_u; i_curr_v]  current image feature (pixels)
% Z_est:  Estimated depth of the point (m)
% f, c_u, c_v: Camera parameters
% Normalized image coordinates i_u, i_v 
% Interaction Matrix L_s (2x6)
clear all;
close all;
clc;

function s = project_3d_to_2d(P_C, K)         % Applies the pinhole model to project a 3D point in the Camera frame (P_C) to a 2D image feature (s)
X = P_C(1);                                   % P_C: [X; Y; Z] vector in camera frame (m)
Y = P_C(2);
Z = P_C(3);
if Z <= 0
    s = [inf; inf];                           % Handle point behind the camera
    return;
end                                     
s_homogeneous = K * (P_C / Z);                % K:   3x3 Intrinsic parameter matrix and Pinhole projection: s = K * P_C_normalized
s = s_homogeneous(1:2);                       % Extract [i_curr_u; i_curr_v]
end

function L_s = compute_interaction_matrix(i_curr, Z_est, f, c_u, c_v)      % COMPUTE Interaction matrix and Computes the 2x6 Image Jacobian (Interaction Matrix) L_s for a point feature
i_curr_u = i_curr(1);
i_curr_v = i_curr(2);
i_u = (i_curr_u - c_u) / f;
i_v = (i_curr_v - c_v) / f;
L_s = [...
    -1/Z_est, 0, i_u/Z_est, i_u*i_v, -(1 + i_u^2), i_v;
    0, -1/Z_est, i_v/Z_est, (1 + i_v^2), -i_u*i_v, -i_u
];
end

% PARAMETERS & INITIALIZATION
% Camera Intrinsic Parameters (Pinhole Model)
% Focal length in pixels (fx = fy = f)
% Principal point i_curr_u-coordinate (image center)
% Principal point i_curr_v-coordinate (image center)

f = 800;   
c_u = 320;  
c_v = 240;   
K = [f, 0, c_u; 
     0, f, c_v; 
     0, 0, 1];

% Control Parameters
LAMBDA = 0.5;   % IBVS control gain (lambda)
DT = 0.01;      % Simulation time step
T_MAX = 9.0;   % Maximum simulation time

% 3D Target Point (fixed in the world frame)
P_TARGET_W = [0.5; 0.1; 3.0]; % Target position (X, Y, Z) in meters

% Initial Camera Pose (W to C transformation, R_WC, t_WC)
R_WC = eye(3);                                  % R_WC: Rotation World to Camera (Identity = no rotation)
t_WC = [1.0; 1.0; 1.0];                         % Initial camera position in meters
s_star = [c_u; c_v];    % Desired 2D Image Feature (Target centered in the image) and (i_curr_u*, i_curr_v*) 
% Data logging initialization
num_steps = floor(T_MAX / DT);
t_history = zeros(1, num_steps);
e_history = zeros(2, num_steps);
s_history = zeros(2, num_steps);
t_WC_history = zeros(3, num_steps);
step = 1;
fprintf('Starting IBVS simulation. Target: [%.2f, %.2f, %.2f]\n', P_TARGET_W(1), P_TARGET_W(2), P_TARGET_W(3));
fprintf('Initial Camera Position: [%.2f, %.2f, %.2f]\n', t_WC(1), t_WC(2), t_WC(3));

% SIMULATION LOOP (Kinematic Model)
for t = 0:DT:(T_MAX - DT)
    t_history(step) = t;
    t_WC_history(:, step) = t_WC;    
    P_C = R_WC * P_TARGET_W + t_WC;     % Compute 3D point in Camera Frame i.e, P_C = R_WC * P_W + t_WC
    i_curr = project_3d_to_2d(P_C, K);  % Compute Current 2D Image Feature (s)
    s_history(:, step) = i_curr;
    e = i_curr - s_star;                % Calculate Feature Error (e)
    e_history(:, step) = e;

    if norm(e) < 1.0                    % % Check for convergence and convergence threshold (1 pixel)
        fprintf('\nConverged at t=%.2fs. Final error: %.2f pixels.\n', t, norm(e));
        break;
    end

    Z_est = P_C(3);                     % Use current depth as the estimate
    L_s = compute_interaction_matrix(i_curr, Z_est, f, c_u, c_v);

    % Compute Control Velocity (v_c) and the basic IBVS control law: v_c = -lambda * L_s+ * e
    L_s_pinv = pinv(L_s);               % Pseudo-inverse
    v_c = -LAMBDA * L_s_pinv * e;
    v_linear_C = v_c(1:3);              % Linear velocity command in Camera frame

    % Update Camera Pose (Simple Kinematic Integration) and Velocity in World Frame: v_linear_W = R_CW * v_linear_C = R_WC' * v_linear_C and
    % also R_WC is kept constant (pure translational control) for simplicity
    v_linear_W = R_WC' * v_linear_C;
    t_WC = t_WC + v_linear_W * DT;
    step = step + 1;
end

% VISUALIZATION OF RESULTS
% Adjust the number of steps recorded
% 'step' holds the value of the next intended step, which is 1 greater than the last recorded index

last_index = step - 1;

% Trim unused data after early termination/full run
t_history = t_history(1:last_index);
e_history = e_history(:, 1:last_index);
s_history = s_history(:, 1:last_index);
t_WC_history = t_WC_history(:, 1:last_index);

% (Rest of the plotting code)
figure('Name', 'IBVS Simulation Results', 'Position', [100, 200, 700, 600]);

% Image Feature Error
subplot(3, 1, 1);
plot(t_history, e_history(1, :), 'LineWidth', 2, 'DisplayName', 'e_u');
hold on;
plot(t_history, e_history(2, :), 'LineWidth', 2, 'DisplayName', 'e_v');
title('Image Feature Error (e)');
ylabel('Error (pixels)');
grid on;
legend show;

% Image Feature Trajectory
subplot(3, 1, 2);
plot(t_history, s_history(1, :), 'LineWidth', 2, 'DisplayName', 'i_curr_u current');
hold on;
plot(t_history, s_history(2, :), 'LineWidth', 2, 'DisplayName', 'i_curr_v current');
yline(s_star(1), 'k--', 'i_curr_u^* desired');
title('Image Features Trajectory (s)');
ylabel('Feature Position (pixels)');
grid on;
legend show;

% Quadrotor (Camera) Position
subplot(3, 1, 3);
plot(t_history, t_WC_history(1, :), 'LineWidth', 2, 'DisplayName', 'X_c');
hold on;
plot(t_history, t_WC_history(2, :), 'LineWidth', 2, 'DisplayName', 'Y_c');
plot(t_history, t_WC_history(3, :), 'LineWidth', 2, 'DisplayName', 'Z_c');
title('Camera Position Trajectory (t_{WC})');
xlabel('Time (s)');
ylabel('Position (m)');
grid on;
legend show;