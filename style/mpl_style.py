# code for enforcing consistent styling in figures
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager


def set_custom_font():

    font_path = "style/dejavu-sans.condensed.ttf"

    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    #  Set it as default matplotlib font
    matplotlib.rc("font", family="sans-serif")
    matplotlib.rcParams.update({"font.sans-serif": prop.get_name()})


def set_custom_props():
    matplotlib.rcParams.update(
        {
            "figure.dpi": 300,
            # Axes
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            # Lines and markers
            "lines.linewidth": 2,
            "lines.markersize": 20,
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
            "legend.borderaxespad": 0.85,
            "legend.handletextpad": 0.5,
            "legend.handlelength": 1.0,
            # Saving figures
            "savefig.dpi": 300,
            "savefig.format": "png",
        }
    )


def apply_style():
    set_custom_font()
    set_custom_props()
    return


apply_style()
print("Custom matplotlib styles have been applied.")
