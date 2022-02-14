% update the Q parameters after each iteration
% c is the choice at last time step, R is the reward at last time step
% R is 1 if the reward is delivered to the side of choice, 0 otherwise
% alpha implements learning, and zeta implements forgetting
function [Q_r, Q_l] = q_update(Q_r, Q_l, c, R, alpha, zeta)
    if c == -1 % choose left
        delta = R - Q_l;
        Q_l = zeta * Q_l + alpha * delta;
    else
        delta = R - Q_r;
        Q_r = zeta * Q_r + alpha * delta;
    end
end
