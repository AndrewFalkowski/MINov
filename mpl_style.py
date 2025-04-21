import matplotlib as mpl
import matplotlib.font_manager as fm


def apply_style():
    """Apply custom matplotlib style parameters."""
    # font_path = "/Users/andrewf/Library/Fonts/dejavu-sans.condensed.ttf"
    # font_path = "/System/Library/Fonts/Supplemental/Academy Engraved LET Fonts.ttf"
    # font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rcParams.update(
        {
            # Figure
            "figure.dpi": 300,
            # Font
            # "font.family": "sans-serif",
            # "font.sans-serif": font_name,
            # "font.size": 9,
            # Axes
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            # Lines and markers
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "lines.markeredgewidth": 0.75,
            # Ticks
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.top": True,
            "xtick.bottom": True,
            "ytick.left": True,
            "ytick.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            # Legend
            "legend.frameon": True,
            "legend.fontsize": 9,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
            "legend.loc": "best",
            # "legend.handletextpad": 0.005,
            "legend.handletextpad": 0.05,
            # "legend.borderaxespad": 0.75
            "legend.borderaxespad": 1.0,
            "legend.handlelength": 1.25,
            # Saving figures
            "savefig.dpi": 300,
            "savefig.format": "png",
        }
    )

    return


apply_style()
