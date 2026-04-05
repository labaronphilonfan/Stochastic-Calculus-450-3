"""
HW1 Problem 3 — Brownian Bridge Simulation

For each sample,simulate one standard Brownian motion B_t on our closed [0,1],
then we derive two other processes from that smae initial Bt path:
    1) B_t         standard Brownian motion
    2) B_t - B_1     BM shifted down by its terminal value
    3) B_t - t * B_1  the Brownian Bridge (X_t from Problem 3), which is our main thing


"""

import numpy as np
import matplotlib.pyplot as plt

#Parameters
N = 1000          # number of time steps (finer = smoother path)
dt = 1.0 / N      # time increment
num_samples = 4   # how many independent BM paths to simulate

# time grid: 0, dt, 2*dt, ..., 1
t = np.linspace(0, 1, N + 1)



# figure setup 
fig, axes = plt.subplots(num_samples, 3, figsize=(16, 3.2 * num_samples),
                         sharex=True, sharey='row')

# Column titles
col_titles = [
    r"$B_t$ (Standard BM)",
    r"$B_t - B_1$ (Shifted to end at 0)",
    r"$B_t - t \cdot B_1$ (Brownian Bridge)"
]

for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=13, fontweight='bold', pad=10)

# Simulation loop
for i in range(num_samples):
    # Step 1: Generate increments dB ~ N(0, dt), then cumsum to get B_t
    # We prepend a 0 so that B_0 = 0.
    increments = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)
    B = np.concatenate([[0], np.cumsum(increments)])

    # Step 2: Read off B_1 (the terminal value)
    B1 = B[-1]

    # Step 3: Compute the two derived processes
    shifted = B - B1          # B_t - B_1:  starts at -B_1, ends at 0
    bridge  = B - t * B1      # B_t - tB_1: starts at 0,    ends at 0

    # ── Plotting ────────────────────────────────────────────────────
    color = f"C{i}"  # matplotlib's automatic color cycle

    # Column 0: Standard BM
    ax0 = axes[i, 0]
    ax0.plot(t, B, color=color, linewidth=0.8)
    ax0.axhline(0, color='gray', linewidth=0.4, linestyle='--')
    ax0.set_ylabel(f"Sample {i+1}", fontsize=11)

    # Mark B_1 with a dot
    ax0.plot(1, B1, 'o', color=color, markersize=5)
    ax0.annotate(f"$B_1 = {B1:.2f}$", xy=(1, B1),
                 xytext=(0.75, B1 + 0.15),
                 fontsize=8, color=color,
                 arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    # Column 1: B_t - B_1
    ax1 = axes[i, 1]
    ax1.plot(t, shifted, color=color, linewidth=0.8)
    ax1.axhline(0, color='gray', linewidth=0.4, linestyle='--')
    # Mark start and end
    ax1.plot(0, shifted[0], 'o', color=color, markersize=5)
    ax1.plot(1, shifted[-1], 'o', color=color, markersize=5)

    # Column 2: Brownian Bridge
    ax2 = axes[i, 2]
    ax2.plot(t, bridge, color=color, linewidth=0.8)
    ax2.axhline(0, color='gray', linewidth=0.4, linestyle='--')
    ax2.plot(0, 0, 'o', color=color, markersize=5)
    ax2.plot(1, 0, 'o', color=color, markersize=5)

# ── Final formatting ───────────────────────────────────────────────
for j in range(3):
    axes[-1, j].set_xlabel("$t$", fontsize=11)

fig.suptitle("From Brownian Motion to the Brownian Bridge",
             fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()
plt.savefig("brownian_bridge_comparison.png", dpi=200, bbox_inches='tight')
plt.show()

print("Saved: brownian_bridge_comparison.png")