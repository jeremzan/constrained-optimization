import numpy as np

def qp(x):
    return x[0]**2 + x[1]**2 + (x[2] + 1)**2, np.array([2*x[0], 2*x[1], 2*x[2] + 2]), np.eye(3) * 2

def ineq_constraint_qp(idx, x):
    f = -x[idx]
    g = np.zeros(3)
    g[idx] = -1
    h = np.zeros((3, 3))
    return f, g, h

starting_point_qp = np.array([0.1, 0.2, 0.7], dtype=np.float64)
eq_constraint_mat_qp = np.array([[1, 1, 1]])
eq_constraint_rhs_qp = np.array([1])

def lp(x):
    return -x[0] - x[1], np.array([-1, -1]), np.zeros((2, 2))

def ineq_constraint_lp(idx, x):
    constraints = [
        (-x[0] - x[1] + 1, np.array([-1, -1])),
        (x[1] - 1, np.array([0, 1])),
        (x[0] - 2, np.array([1, 0])),
        (-x[1], np.array([0, -1]))
    ]
    f, g = constraints[idx]
    h = np.zeros((2, 2))
    return f, g, h

starting_point_lp = np.array([0.5, 0.75], dtype=np.float64)
eq_constraint_mat_lp = np.empty((0, 2))
eq_constraint_rhs_lp = np.empty((0, 2))
