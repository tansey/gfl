import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np

def load_grid(filename):
    grid = np.loadtxt(filename, delimiter=',')
    dim1 = int(np.sqrt(grid.shape[0]))
    return grid.reshape((dim1, dim1))

truth = load_grid('../example/truth.csv')
data = load_grid('../example/data.csv')
output = load_grid('../example/output.csv')

fig, ax = plt.subplots(1,3)
im = ax[0].imshow(truth, interpolation='none', vmin=-2, vmax=2, cmap='gray')
ax[1].imshow(data, interpolation='none', vmin=-2, vmax=2, cmap='gray')
ax[2].imshow(output, interpolation='none', vmin=-2, vmax=2, cmap='gray')

ax[0].set_title('Truth')
ax[1].set_title('Data')
ax[2].set_title('Output')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, ticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
plt.savefig('../example/result.pdf')