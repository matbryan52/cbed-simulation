from typing import TYPE_CHECKING
import numpy as np
from .utils import to_miller_ltx

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def plot_pattern(
    fig: 'Figure',
    ax: 'Axes',
    frame: np.ndarray,
    max_extent,
    spots,
    intensity,
    millers,
    scatter_alpha=1.,
    interactive: bool = True,
):
    miller_labels = [to_miller_ltx(*m) for m in millers]

    if frame is not None:
        img_kwargs = dict(cmap='grey')
        if max_extent is not None:
            img_kwargs['extent'] = (-max_extent, max_extent, -max_extent, max_extent)
        ax.imshow(
            frame,
            **img_kwargs,
        )
    sc = ax.scatter(
        spots[:, 0],
        spots[:, 1],
        s=np.sqrt(intensity) * 1.1,
        facecolor='none',
        edgecolor='red',
        alpha=scatter_alpha,
    )
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(5, 5),
        textcoords="offset points",
        bbox=dict(fc="w", alpha=0.5),
    )
    annot.set_visible(False)

    def update_annot(ind):
        idx = ind["ind"][0]
        pos = sc.get_offsets()[idx]
        annot.xy = pos
        text = miller_labels[idx]
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    if interactive:
        fig.canvas.mpl_connect("motion_notify_event", hover)
