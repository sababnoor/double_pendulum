import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Parameters
g = 9.81
L1 = 1
L2 = 1
m1 = 1
m2 = 1

# Equations of motion
def deriv(t, y):
    theta1, z1, theta2, z2 = y

    c = np.cos(theta1 - theta2)
    s = np.sin(theta1 - theta2)

    theta1dot = z1
    theta2dot = z2

    z1dot = (
        m2 * g * np.sin(theta2) * c
        - m2 * s * (L1 * z1**2 * c + L2 * z2**2)
        - (m1 + m2) * g * np.sin(theta1)
    ) / (L1 * (m1 + m2 * s**2))

    z2dot = (
        (m1 + m2)
        * (
            L1 * z1**2 * s
            - g * np.sin(theta2)
            + g * np.sin(theta1) * c
        )
        + m2 * L2 * z2**2 * s * c
    ) / (L2 * (m1 + m2 * s**2))

    return [theta1dot, z1dot, theta2dot, z2dot]

# Initial conditions
y0 = [np.pi / 2, 0, np.pi / 2 + 0.01, 0]

t = np.linspace(0, 30, 1000)

sol = solve_ivp(
    deriv,
    [t[0], t[-1]],
    y0,
    t_eval=t
)

theta1 = sol.y[0]
theta2 = sol.y[2]

x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)

x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-2.2,2.2)
ax.set_ylim(-2.2,2.2)
ax.set_aspect('equal')
ax.set_facecolor("black")

line, = ax.plot([], [], lw=2, color="white")
trail, = ax.plot([], [], lw=1, color="cyan")

trail_x = []
trail_y = []

def update(frame):
    line.set_data(
        [0, x1[frame], x2[frame]],
        [0, y1[frame], y2[frame]]
    )

    trail_x.append(x2[frame])
    trail_y.append(y2[frame])

    trail.set_data(trail_x, trail_y)

    return line, trail

ani = FuncAnimation(
    fig,
    update,
    frames=len(t),
    interval=20,
    blit=True
)

plt.show()
