import numpy as np

dt = 0.001
T = 5.0
N = int(T / dt)

t = np.linspace(0, T, N + 1)

# th1 = np.linspace(0, np.pi, N+1)
th1 = np.pi / 2.0 * (np.tanh(10 * t / T - 5) + 1)
th2 = np.linspace(0, 0, N + 1)

th1_vel = np.diff(th1, prepend=th1[0]) / dt
th2_vel = np.diff(th2, prepend=th2[0]) / dt

th1_acc = np.diff(th1_vel, prepend=th1_vel[0]) / dt
th2_acc = np.diff(th2_vel, prepend=th2_vel[0]) / dt

data = np.asarray([t, th1, th2, th1_vel, th2_vel, th1_acc, th2_acc]).T
np.savetxt(
    "trajectory_modelfree.csv",
    data,
    delimiter=",",
    header="time,pos1,pos2,vel1,vel2,acc1,acc2",
    comments="",
)
