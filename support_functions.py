import numpy as np

def cost_f(s, data):
    X = s.X  # Add normalising over increment number
    y = np.zeroes(0,6)
    w = get_combined_workload(X, s.c, data.w, data.c_matrix, data.d_matrix, data.p_matrix, data.alpha, data.T)
    y[0] = sum(w) / sum(data.h)
    y[1] = unbalanced_workload(w, data.h)
    y[2] = staff_dissatisfaction(X, data.P, 1,)
    y[3] = staff_dissatisfaction(X, data.P, 2)
    y[4] = average_staff_per_module(X)
    y[5] = peak_load(X, s.C, data.h, data.c_matrix, data.d_matrix, data.p_matrix, data.t_matrix, data.alpha,data.T)
    y[6] = variation_from_previous_year_teach(X, data.R/100)
    # Constraints go here if using them
    return y, w


def average_staff_per_module(X):
    return np.count_nonzero(X) / len(X)


def get_combined_workload(X, C, w, c_matrix, d_matrix, p_matrix, alpha, T):
    return np.multiply(c_matrix, C) + np.multiply((d_matrix + np.multiply((1+alpha)*T, p_matrix)), X)


def peak_load(X, C, h, c_matrix,d_matrix, p_matrix, t_matrix, alpha, T):
    temp = get_combined_workload(X, C, w, c_matrix, d_matrix, p_matrix, alpha, T)
    return max(abs(np.divide(
        sum(np.multiply(temp, np.where(t_matrix == 1)) - np.multiply(temp, np.where(t_matrix == 2)), h))))


def staff_dissatisfaction(X, P, level):
    return max(sum(X[np.argwhere(P >= level)]))


def unbalanced_workload(w, h):
    div = np.divide(w, h)
    return max(div) - min(div)


def variation_from_previous_year_teach(X, X_old):
    return sum(sum(abs(X - X_old)))
