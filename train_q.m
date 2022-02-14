% choices are uniformly encoded as: -1 for left, 1 for right

% find the parameters with gradient descent
% a is the learning rate
% epochs is the number of iterations we want gradient descent to run for
function [beta, b, mse] = train_q(beta, b, trials, a)
    for i = 1:4000
        beta_g = beta_gradient(beta, b, trials);
        b_g = b_gradient(beta, b, trials);
        beta = beta - a * beta_g;
        b = b - a * b_g;
    end
    mse = MSE(beta, b, trials);
end


% beta determines the balance between exploration and exploitation
% b is a bias term, and kappa implements auto correlation with previous
% choice
% a_pre is the action taken in the last time step, 1 for right and -1 for
% left
function [right_prob] = q_predict(Q_relative, beta, b)%, kappa, a_pre)
    right_prob = 1 / (1 + exp(-beta * Q_relative + b));% + kappa * a_pre));
end


% trails: n by 4 data (Q_r-Q_l, p_r_hat, c, R) 

function mse = MSE(beta, b, trails)
    mse = 0;
    n = size(trails, 1);
    for i = 2:n
        Q_relative = trails(i, 1);
        p_r_hat = trails(i, 2);
        % a_pre = trails(i-1, 4);
        P = q_predict(Q_relative, beta, b);
        mse = mse + (1/n) * (p_r_hat - P) ^ 2;
    end
end

% gradient for beta
function beta_g = beta_gradient(beta, b, trails)
    beta_g = 0;
    n = size(trails, 1);
    for i = 2:n
        Q_relative = trails(i, 1);
        p_r_hat = trails(i, 2);
        P = q_predict(Q_relative, beta, b);
        beta_g = beta_g + (-2 / n) * (p_r_hat - P) * P * (P - 1) * (-Q_relative);
    end
end

% gradient for b
function b_g = b_gradient(beta, b, trails)
    b_g = 0;
    n = size(trails, 1);
    for i = 2:n
        Q_relative = trails(i, 1);
        p_r_hat = trails(i, 2);
        P = q_predict(Q_relative, beta, b);
        b_g = b_g + (-2 / n) * (p_r_hat - P) * P * (P - 1);
    end
end

% % gradient for kappa
% function kappa_g = kappa_gradient(beta, b, kappa, trails)
%     kappa_g = 0;
%     n = size(trails, 1);
%     for i = 2:n
%         Q_l = trails(i, 1);
%         Q_r = trails(i, 2);
%         p_r_hat = trails(i, 3);
%         a_pre = trails(i-1, 4);
%         P = q_predict(Q_r, Q_l, beta, b, kappa, a_pre);
%         kappa_g = kappa_g + (-2 / n) * (p_r_hat - P) * P * (P - 1) * a_pre;
%     end
% end
