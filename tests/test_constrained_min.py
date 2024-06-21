import sys
import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.constrained_min import InteriorPointMinimizer
from examples import (qp,
                      lp, 
                      ineq_constraint_qp, 
                      ineq_constraint_lp, 
                      eq_constraint_mat_qp, 
                      eq_constraint_rhs_qp, 
                      eq_constraint_mat_lp, 
                      eq_constraint_rhs_lp, 
                      starting_point_qp, 
                      starting_point_lp)

class TestInteriorPointMinimizer(unittest.TestCase):

    minimizer = InteriorPointMinimizer()

    def plot(self, title, path_points=None, obj_values_1=None, obj_values_2=None, label_1=None, label_2=None, is_3d=False, final_point=None):
        plt.style.use('ggplot')  # Use ggplot style
        color_1 = '#1f77b4'  # Blue
        color_2 = '#ff7f0e'  # Orange
        path_color = '#2ca02c'  # Green
        final_point_color = '#d62728'  # Red
        feasible_region_color = '#8c564b'  # Brown

        if obj_values_1 is not None or obj_values_2 is not None:
            fig, ax = plt.subplots()
            if obj_values_1 is not None:
                ax.plot(obj_values_1, label=label_1, color=color_1)
            if obj_values_2 is not None:
                ax.plot(obj_values_2, label=label_2, color=color_2)
            ax.legend()
            ax.set(title=title, xlabel="# iterations", ylabel="Objective function value")
            plt.show()
        
        if path_points is not None:
            if is_3d:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                path = np.array(path_points)
                ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color=feasible_region_color, alpha=0.5)
                ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path', color=path_color)
                if final_point is not None:
                    ax.scatter(*final_point, s=50, c=final_point_color, marker='o', label='Final candidate')
                ax.set(title="Feasible Regions and Path", xlabel='x', ylabel='y', zlabel='z')
                plt.legend()
                ax.view_init(45, 45)
                plt.show()
            else:
                d = np.linspace(-2, 4, 300)
                x, y = np.meshgrid(d, d)
                plt.imshow(((y >= -x + 1) & (y <= 1) & (x <= 2) & (y >= 0)).astype(int), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.3)
                x = np.linspace(0, 4, 2000)
                y1, y2, y3 = -x + 1, np.ones(x.size), np.zeros(x.size)
                plt.plot(x, y1, label='y = -x + 1', color=color_1)
                plt.plot(x, y2, label='y = 1', color=color_2)
                plt.plot(x, y3, label='y = 0', color=feasible_region_color)
                plt.plot(np.ones(x.size) * 2, x, label='x = 2', color='purple')
                x_path, y_path = zip(*path_points)
                plt.plot(x_path, y_path, label="algorithm's path", color=path_color, marker=".", linestyle="--")
                if final_point is not None:
                    plt.scatter(*final_point, s=50, c=final_point_color, marker='o', label='Final candidate')
                plt.xlim(0, 3)
                plt.ylim(0, 2)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                plt.xlabel(r"$x$")
                plt.ylabel(r"$y$")
                plt.suptitle('Feasible region and path 2D')
                plt.show()




    def run_test(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, starting_point, is_3d):
        x_s, obj_values, outer_x_s, outer_obj_values = self.minimizer.interior_pt(
            func,
            ineq_constraints,
            eq_constraints_mat,
            eq_constraints_rhs,
            starting_point
        )

        final_point = x_s[-1]
        func_name = func.__name__.upper()
        print("\n")
        print(f"Running test for {func_name}")
        print("Final candidate: ", "[" , ", ".join(f"{coord:.4f}" for coord in final_point), "]")
        print(f"Objective value at final candidate: {func(final_point)[0]:.4f}")
        for i, ineq in enumerate(ineq_constraints, start=1):
            print(f"Ineq constraint {i} value at final candidate: {ineq(final_point)[0]:.4f}")

        self.plot(title=f"Objective function values of {func_name} function",
             obj_values_1=outer_obj_values,
             obj_values_2=obj_values,
             label_1="Outer objective values",
             label_2="Objective values")
        
        self.plot(title="Feasible Regions and Path", path_points=x_s, is_3d=is_3d, final_point=final_point)

    def test_qp(self):
        self.run_test(
            qp,
            [lambda x: ineq_constraint_qp(0, x), lambda x: ineq_constraint_qp(1, x), lambda x: ineq_constraint_qp(2, x)],
            eq_constraint_mat_qp,
            eq_constraint_rhs_qp,
            starting_point_qp,
            is_3d=True
        )

    def test_lp(self):
        self.run_test(
            lp,
            [lambda x: ineq_constraint_lp(0, x), lambda x: ineq_constraint_lp(1, x), lambda x: ineq_constraint_lp(2, x), lambda x: ineq_constraint_lp(3, x)],
            eq_constraint_mat_lp,
            eq_constraint_rhs_lp,
            starting_point_lp,
            is_3d=False
        )

if __name__ == "__main__":
    unittest.main()
