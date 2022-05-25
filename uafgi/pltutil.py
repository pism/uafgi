import matplotlib.colors
import matplotlib.cm

def plot_cbar(fig, cmap, vmin, vmax, orientation):
    """Plots a "raw" colorbar, independent of any plot."""

    # Set up mapping between vel_year and color
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    ax = fig.add_axes((.1,.05,.85,.5))
    cbar = fig.colorbar(mapper, ax=ax, orientation=orientation)
    ax.remove()   # https://stackoverflow.com/questions/40813148/save-colorbar-for-scatter-plot-separately
    return cbar
