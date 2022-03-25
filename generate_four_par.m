function [Q_l, Q_r, zeta, alpha, beta, b] = generate_four_par(trialresponseside, trialreward, lr, name)
    % searching for the best Q learning update parameter
    min_mse = 100000;
    best_zeta = 0;
    best_alpha = 0;
    best_beta = 0;
    best_b = 0;

    % save a copy of trialresponseside and trialreward then removes the
    % nan values from them for the parameter search
    trialresponseside_original = trialresponseside;
    trialreward_original = trialreward;
    trialresponseside(isnan(trialresponseside)) = [];
    trialreward(isnan(trialreward)) = [];
    % set all trials where the mouse choose left to 0 to get rightward prob
    trialresponseside(trialresponseside == -1) = 0;
    s = length(trialresponseside);
    smoothed = movmean(trialresponseside, 21);
    f = waitbar(0, 'traning', 'Name', char(name));

    % grid search for zeta and alpha values that produces the smallest MSE
    % with the actual action of the mouse
    for zeta = [0:0.01:1]
        for alpha = [0:0.01:1]
            waitbar(zeta + alpha*0.01, f, join(['calculated ', string(zeta*100+alpha), '%']))
            % produce the fictitious Q values using the update function
            q_l = 0;
            q_r = 0;

            s = length(trialresponseside);
            relative = zeros(s, 1);
            Q_l = zeros(s, 1);
            Q_r = zeros(s, 1);

            for i = 2:s
                if trialresponseside(i-1) == 1 % right
                    q_r = zeta * q_r + alpha * (trialreward(i-1) - q_r);
                    q_l = zeta * q_l;
                else
                    q_l = zeta * q_l + alpha * (trialreward(i-1) - q_l);
                    q_r = zeta * q_r;
                end
                relative(i) = q_r - q_l;
                Q_l(i) = q_l;
                Q_r(i) = q_r;
            end

            results = zeros(10, 3);
            trials = zeros(s, 2);
            trials(:, 1) = relative';
            trials(:, 2) = smoothed';

            % using the relative values produced for gradient descent
            % initialize with 10 set of random values for beta and b to avoid
            % possible local minima
            counter = 10;
            while counter > 0
                beta = randi([-10, 10]);
                b = randi([-10, 10]);
                [beta, b, MSE] = train_q(beta, b, trials, lr);
                results(counter, 1) = beta;
                results(counter, 2) = b;
                results(counter, 3) = MSE;
                counter = counter - 1;
            end
            % find the set of parameters that gives the smallest mse
            MSEs = results(:, 3);
            [MSE, best_param_loc] = min(MSEs');
            mse = MSE;
            beta = results(best_param_loc, 1);
            b = results(best_param_loc, 2);
            % find the set of parameters that gives the smallest mse
            if mse < min_mse
                min_mse = mse;
                best_zeta = zeta;
                best_alpha = alpha;
                best_beta = beta;
                best_b = b;
            end
        end
    end

    zeta = best_zeta;
    alpha = best_alpha;
    beta = best_beta;
    b = best_b;

    % with all parameters found, we can run the generative model using the
    % original data (with the nan values)
    trialreward = trialreward_original;
    trialreward(isnan(trialreward_original)) = 0;
    trialresponseside = trialresponseside_original;
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
    smoothed = movmean(trialresponseside_original, 21, 'omitnan');

    close(f)

    figure
    plot([1:s], choices, [1:s], smoothed, [1:s], reward);
    ylabel('probability of rightward choice')
    xlabel('trials')
    legend('generative model', 'mouse action', 'reward')
    title(join(['zeta: ', string(zeta), 'alpha: ', string(alpha), 'MSE: ', string(min_mse)]))
    saveas(gcf, char(fullfile('results', name)), 'jpg')
end