"""idm.tools.plotting — shared CMS-style plotting helpers.

Encodes the CMS / mplhep plotting conventions so plots across the analysis are consistent
and publication-ready: ``figsize=(10, 10)``, the CMS stylesheet, an mplhep ``histplot``
wrapper that makes you pass ``yerr`` explicitly for derived (non-count) quantities (mplhep
otherwise draws bogus sqrt(N) bars), a CMS ``exp_label``, and a save helper that writes both
PDF and PNG with ``bbox_inches='tight'`` and ``dpi=200``.

Lint plotting scripts against these conventions with the local ``conventions/lint_plots.py``.
Conventions adapted from the JFC framework (violatingcp/jfc).
"""

import matplotlib.pyplot as plt
import mplhep as hep


def use_cms_style():
    """Apply the CMS mplhep stylesheet (call once per session/script)."""
    hep.style.use("CMS")


def new_figure(ratio=False):
    """Create a CMS-sized figure.

    ratio=False -> a single (10, 10) panel, returns (fig, ax).
    ratio=True  -> a main + ratio panel sharing the x-axis (no gap), returns (fig, ax, rax).
    """
    if ratio:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        fig.subplots_adjust(hspace=0)
        return fig, ax, rax
    fig, ax = plt.subplots(figsize=(10, 10))
    return fig, ax


def plot_hist(h, ax, yerr=None, **kwargs):
    """Draw a histogram with mplhep histplot.

    For a DERIVED quantity (a hist filled via ``.view()[:] = values`` rather than counts —
    e.g. an efficiency, ratio, or scale factor) pass ``yerr=`` explicitly; otherwise mplhep
    draws sqrt(bin_content) error bars, which are meaningless for non-count values.
    """
    return hep.histplot(h, ax=ax, yerr=yerr, **kwargs)


def cms_label(ax, data=False, com=13.6, lumi=None, label="Preliminary"):
    """Add the CMS label to the main panel only (com=13.6 TeV for Run 3)."""
    hep.cms.label(label, data=data, com=com, lumi=lumi, ax=ax)


def save(fig, path_noext, dpi=200):
    """Save the figure as both PDF and PNG (bbox_inches='tight')."""
    fig.savefig(f"{path_noext}.pdf", bbox_inches="tight", dpi=dpi)
    fig.savefig(f"{path_noext}.png", bbox_inches="tight", dpi=dpi)
