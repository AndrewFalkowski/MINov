import matplotlib as mpl
import matplotlib.pyplot as plt
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
            "font.size": 9,
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
            "legend.handletextpad": 0.05,
            "legend.borderaxespad": 0.65,
            "legend.handlelength": 1.25,
            # Saving figures
            "savefig.dpi": 300,
            "savefig.format": "png",
        }
    )

    return


fpath = Path("/Users/andrewf/Library/Fonts/dejavu-sans.condensed.ttf")
mpl.font_manager.fontManager.addfont(fpath)
if fpath.exists():
    # Add the font to the font manager
    mpl.font_manager.fontManager.addfont(str(fpath))

    # Create font properties object
    prop = mpl.font_manager.FontProperties(fname=str(fpath))
    font_name = prop.get_name()

    # Set as default
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_name]

    # Set specific text properties
    # For newer matplotlib versions
    mpl.rc("font", **{"family": "sans-serif", "sans-serif": [font_name]})

    def apply_condensed_font(fig):
        for ax in fig.axes:
            # Apply to all text elements on the axes
            for text in ax.texts:
                text.set_fontproperties(prop)

            # Apply to title and axis labels
            ax.title.set_fontproperties(prop)
            ax.xaxis.label.set_fontproperties(prop)
            ax.yaxis.label.set_fontproperties(prop)

            # Apply to tick labels
            for label in ax.get_xticklabels():
                label.set_fontproperties(prop)
            for label in ax.get_yticklabels():
                label.set_fontproperties(prop)

            # Apply to legend if it exists
            legend = ax.get_legend()
            if legend:
                # Apply to legend title
                if legend.get_title():
                    legend.get_title().set_fontproperties(prop)

                # Apply to legend text
                for text in legend.get_texts():
                    text.set_fontproperties(prop)


if __name__ == "__main__":
    apply_style()
