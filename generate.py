import numpy as np
import scipy.stats


def generate(zeta, alpha, beta, b, trialrsponseside, trialreward, leftP, rightP):
    ntrial = len(leftP)
    Q_l = [0]
    Q_r = [0]
    choices_sum = [0] * len(trialrsponseside)
    for l in range(100):
        choices = []
        for i in range(ntrial):
            p_r = 1 / (1 + np.exp(-beta * (Q_r[i] - Q_l[i]) + b))
            c = np.random.binomial(n=1, p=p_r)
            if c == 1:
                rewarded = np.random.binomial(n=1, p=rightP[i])
                choices.append(1)
            else:
                rewarded = np.random.binomial(n=1, p=leftP[i])
                choices.append(-1)

            if c == 1:
                delta = rewarded - Q_r[i]
                Q_r.append(zeta * Q_r[i] + alpha * delta)
                Q_l.append(zeta * Q_l[i])
            else:
                delta = rewarded - Q_l[i]
                Q_l.append(zeta * Q_l[i] + alpha * delta)
                Q_r.append(zeta * Q_r[i])
        choices_sum = [sum(x) for x in zip(choices, choices_sum)]

    return np.array(choices_sum) / 100
