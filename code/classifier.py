import numpy as np
import scipy
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.special import beta as beta_fun
from scipy.special import betainc as i_beta_fun
from scipy.stats import beta as beta_distr


def cal_param(sigma, gamma):
    beta = gamma / (1 - sigma) / (1 - gamma)
    alpha = sigma * beta
    return alpha, beta


def beta_auc(alpha, beta):
    x = np.linspace(0, 1, 1000)
    y = beta_distr(beta, alpha).cdf(x)
    z = beta_distr(alpha, beta).cdf(x)
    return scipy.integrate.trapz(z, y)


def beta_accuracy(alpha, beta):
    return beta_distr.cdf(0.5, alpha, beta)


def beta_px(alpha, beta):
    def f(x):
        return 0.5 * (beta_distr(alpha, beta).pdf(x) + beta_distr(beta, alpha).pdf(x))
    return f


def beta_p1(alpha, beta):
        def f(x):
            return 1 / (1 + np.power((1 - x) / x, beta - alpha))
        return f


def beta_p0(alpha, beta):
    return beta_p1(beta, alpha)


# expected mean error

def beta_eme(alpha, beta):
    b = beta_fun(1 + alpha, beta) / beta_fun(alpha, beta)
    return b * np.abs(-i_beta_fun(1 + alpha, beta, 0.5) + i_beta_fun(beta, 1 + alpha, 0.5))


def _accuracy(gamma=0.5, b=10):
    if gamma is not None:
        def f(s):
            a, b = cal_param(s, gamma)
            return beta_distr.cdf(0.5, a, b)
    else:
        def f(s):
            return beta_distr.cdf(0.5, s * b, b)
    return f


_cache = {}


def s_from_accuracy(cal=0.5, b=10):
    if b not in _cache:
        s = np.linspace(0.0001, 0.9999, 1000)
        # f = np.vectorize(_accuracy(b))
        f = _accuracy(cal, b)
        _cache[b] = interp1d(f(s), s)
    return _cache[b]


def pr_beta(sigma=None, gamma=0.5, accuracy=None, size=1, b=10, sigma_cal=None, **kwargs):
    if sigma is None:
        if sigma_cal is None:
            sigma = s_from_accuracy(gamma, b)(accuracy)
        else:
            sigma = sigma_(sigma_cal, gamma)
    if gamma is not None:
        a, b = cal_param(sigma, gamma)
    else:
        a = sigma * b
        gamma = (b - a) / (b - a - 1)
    # if accuracy is None:
    #     accuracy = beta_accuracy(a, b)

    if sigma == 0:  # accuracy == 1.0
        ps = [0] * size
    elif sigma == 1:  # accuracy == 0.5
        ps = [0.5] * size
    else:
        ps = beta_distr(a, b).rvs(size=size)

    return ps, sigma, gamma


f_auc = None


def sigma_(sigma_cal, gamma):
    global f_auc
    if not f_auc:
        import os
        aucs_path = os.path.join(os.path.dirname(__file__), 'aucs.npy')
        aucs = np.load(aucs_path)
        xs = np.linspace(0.01, 0.99, 100)
        ss, gs = np.meshgrid(xs, xs)
        f_auc = scipy.interpolate.interp2d(ss, gs, aucs.T)
    # print(sigma_cal, gamma)

    def f_(x):
        return f_auc(x, gamma) - f_auc(sigma_cal, 0.5)

    try:
        return scipy.optimize.brentq(f_, 0, 1)
    except ValueError:
        return None


def implicit_sg(sigma_cal, gs=np.linspace(0.05, 0.95, 19)):
    ss = [sigma_(sigma_cal, g) for g in gs]
    return np.array([(s, g) for s, g in zip(ss, gs) if s is not None])


def classifier_output(realization, sigma=None, gamma=0.5, accuracy=None, **kwargs):
    size = len(realization)
    error, *val = pr_beta(sigma=sigma, gamma=gamma, accuracy=accuracy, size=size, **kwargs)
    return (tuple([(1 - x if r else x) for r, x in zip(realization, error)]),) + tuple(val)
