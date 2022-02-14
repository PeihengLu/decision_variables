function [choices] = generate_p_values(mouse_code, date)
    filename = join(['AKED', mouse_code, '2021', date, '.mat']);
    load(fullfile("alltaskinfo/", filename));
    load(fullfile("results_data/", filename));
    % make sure matlab doesn't recognize them as the functions with same
    % name
    beta = beta;
    alpha = alpha;
    zeta = zeta;

    trialreward(isnan(trialreward)) = 0;
    s = length(trialreward);
    Q_l = zeros(s, 1);
    Q_r = zeros(s, 1);
    reward = zeros(s, 1);
    choices = zeros(s, 1);

    for i = 1:s
        if trialreward(i) == 1
            reward(i) = trialresponseside(i);
        else
            if isnan(trialresponseside(i))
                reward(i) = 0;
            else
                reward(i) = -trialresponseside(i);
            end
        end
    end

    for i = 1:s
        p_r = q_predict(Q_r(i) - Q_l(i), beta, b);
        % p_r possibility of choosing right
        c = binornd(1, p_r);
        if c == 0
            c = -1;
        end
        choices(i) = c;

        if reward(i) == c
            r = 1;
        else
            r = 0;
        end

        [Q_r(i+1), Q_l(i+1)] = q_update(Q_r(i), Q_l(i), c, r, alpha, zeta);
    end

    choices = movmean(choices, 21);
    reward = movmean(reward, 21);
    smoothed = movmean(trialresponseside, 21, 'omitnan');

    figure
    plot([1:s], choices, [1:s], smoothed, [1:s], reward);
    ylabel('probability of rightward choice')
    xlabel('trials')
    legend('generative model', 'mouse action', 'reward')
    title(join(['zeta: ', string(zeta), 'alpha: ', string(alpha), 'bias: ', string(b)]));
    % saveas(gcf, char(fullfile('results', name)), 'jpg')
end