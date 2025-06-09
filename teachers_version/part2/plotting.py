import matplotlib.pyplot as plt
import numpy as np

def plot_all(solver_output, X_r, X):
    if(len(solver_output) > 0):
        plt.figure()
        plt.title("Solver Output")
        plt.plot(solver_output, label="qd_r")
    if(len(X) > 0):
        plt.figure()
        plt.title("Robot pose")
        X = np.array(X)
        plt.plot(X[:,0], label="x")
        plt.plot(X[:,1], label="y")
        plt.plot(X[:,2], label="theta")
        plt.legend()
    if(len(X_r) > 0):
        X_r = np.array(X_r)
        plt.plot(X_r[:,0],'--', label="x_r")
        plt.plot(X_r[:,1],'--', label="y_r")
        plt.plot(X_r[:,2],'--', label="theta_r")
    plt.show()