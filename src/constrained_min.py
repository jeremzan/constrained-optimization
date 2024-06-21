import numpy as np
import math

class InteriorPointMinimizer:

    def calc_alpha(self, f, p, x, wolfe_constant=0.01, backtrack_constant=0.5, initial_alpha=1):
        alpha = initial_alpha
        f_x, g_x, _ = f(x)
        while f(x + alpha * p)[0] > f_x + wolfe_constant * alpha * np.dot(g_x.T, p):
            alpha *= backtrack_constant
        return alpha

    def phi(self, ineq_constraints, x):
        return_f ,return_g, return_h = 0, 0 ,0
        for func in ineq_constraints:
            f_x, g_x, h_x = func(x)
            return_f += math.log(-f_x)
            g = g_x / f_x
            return_g += g
            g_mesh = np.tile(
                g.reshape(g.shape[0], -1), (1, g.shape[0])
            ) * np.tile(g.reshape(g.shape[0], -1).T, (g.shape[0], 1))
            return_h += (h_x * f_x - g_mesh) / f_x**2

        return -return_f, -return_g, -return_h

    def check_tolerance_conditions(self, iteration_count, current_x, previous_x, lambda_value, param_tol, obj_tol):
        param_condition = iteration_count != 0 and np.sum(np.abs(current_x - previous_x)) < param_tol
        obj_condition = 0.5 * (lambda_value ** 2) < obj_tol
        return param_condition, obj_condition

    def solve_system_and_compute_lambda(self, matrix_block, equation_vector, hessian_matrix, vector_length):
        solution_vector = np.linalg.solve(matrix_block, equation_vector)[:vector_length]
        lambda_value = np.sqrt(np.dot(solution_vector.T, np.dot(hessian_matrix, solution_vector)))
        return solution_vector, lambda_value

    def interior_pt(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
        obj_tol, param_tol, max_iter_inner, max_iter_outer = 1e-6, 1e-6, 100, 50
        epsilon, t, mu = 1e-8, 1, 10

        x = x0
        f_x, g_x, h_x = func(x)
        f_x_phi, g_x_phi, h_x_phi = self.phi(ineq_constraints, x)

        x_s, obj_values = [x0], [f_x]
        outer_x_s, outer_obj_values = [x0], [f_x]

        f_x, g_x, h_x = t * f_x + f_x_phi, t * g_x + g_x_phi, t * h_x + h_x_phi

        for _ in range(max_iter_outer):
            if eq_constraints_mat.size:
                size_zeros = (eq_constraints_mat.shape[0], eq_constraints_mat.shape[0])
                upper_block = np.hstack([h_x, eq_constraints_mat.T])
                lower_block = np.hstack([eq_constraints_mat, np.zeros(size_zeros)])
                block_matrix = np.vstack([upper_block, lower_block])
            else:
                block_matrix = h_x

            eq_vec = np.hstack([-g_x, np.zeros(eq_constraints_mat.shape[0])])
            x_prev, f_prev = x, f_x

            for i in range(max_iter_inner):

                p, lamb = self.solve_system_and_compute_lambda(block_matrix, eq_vec, h_x, len(x))
                param_condition, obj_condition = self.check_tolerance_conditions(i, x, x_prev, lamb, param_tol, obj_tol)

                if param_condition or obj_condition:
                    break
                
                if i != 0 and (f_prev - f_x < obj_tol):
                    break

                alpha = self.calc_alpha(func, p, x)

                x_prev = x
                f_prev = f_x

                x = x + alpha * p
                f_x, g_x, h_x = func(x)
                f_x_phi, g_x_phi, h_x_phi = self.phi(ineq_constraints, x)

                x_s.append(x)
                obj_values.append(f_x)

                f_x, g_x, h_x = t * f_x + f_x_phi, t * g_x + g_x_phi, t * h_x + h_x_phi

            outer_x_s.append(x)
            outer_obj_values.append((f_x - f_x_phi) / t)

            if len(ineq_constraints) / t < epsilon:
                break

            t = mu * t

        return x_s, obj_values, outer_x_s, outer_obj_values
