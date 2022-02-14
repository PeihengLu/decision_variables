function [right_prob] = q_predict(Q_relative, beta, b)%, kappa, a_pre)
    right_prob = 1 / (1 + exp(-beta * Q_relative + b));% + kappa * a_pre));
end