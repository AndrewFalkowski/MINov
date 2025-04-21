# Add the font file
fpath = Path("/Users/andrewf/Library/Fonts/dejavu-sans.condensed.ttf")
mpl.font_manager.fontManager.addfont(fpath)
if fpath.exists():
    # Add the font to the font manager
    mpl.font_manager.fontManager.addfont(str(fpath))

    # Create font properties object
    prop = mpl.font_manager.FontProperties(fname=str(fpath), size=9)
    anno_prop = mpl.font_manager.FontProperties(fname=str(fpath), size=6)
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
                text.set_fontproperties(anno_prop)

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
