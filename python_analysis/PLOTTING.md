## iDMe python_analysis: plotting style

Applies only in the iDMe repo's `python_analysis/` (paths below are relative to it). Apply this
house style to every new or edited plot/plotting script without being asked, so plots are
consistent and slide-readable.

### 1. Helpers: `analysisTools/plotTools.py`

`import analysisTools.plotTools as ptools`. Never copy helpers into a script; new generic helpers
go in `plotTools.py`, but check with the maintainers before changing it.

- `ptools.set_plot_style(**overrides)`: call once, after imports. Applies `PLOT_RCPARAMS` (font 24,
  title 32, titlepad 44 for CMS-label room, axis labels 34, ticks 30, legend 24). Overrides are
  rcParams keys, e.g. `**{'legend.fontsize': 28}`.
- `ptools.new_fig(year, cms_label=True, lumi=None, kind='sim')` creates every figure: one axes,
  `FIGSIZE` (16, 12), CMS label at `CMS_LABEL_FS`=26.
  - Label: italic "Private work (CMS simulation)" upper left, no bold "CMS"; "`<year>` (13.6 TeV)"
    upper right, always above the frame.
  - `kind`: `'sim'` (default) → (CMS simulation); `'data/sim'` → (CMS data/simulation), for data
    + MC on one plot; `'data'` → (CMS data), for data-only.
  - Placement (CMS recommendation): "Private work (...)" inside the frame at top left, unless it
    would cover data or legend, then above the frame. `add_cms_label` does this automatically at
    draw/save (`_CMSLabelPlacer` via `ptools.bbox_covers_data`; 2D hists count only filled bins).
    Never hard-code or hand-move it.
  - Hand-added colorbar: pass `cms_label=False`, then call `ptools.add_cms_label(ax, year,
    kind=...)` after the colorbar. Older `plot_*` functions read kind from optional
    `plot_dict['cms_kind']`.
- `ptools.legend_headroom(ax, leg)`: raises the y-axis top (linear or log) so data sits below an
  upper legend. `ptools.place_legend(ax, **kw)`: legend at upper right, headroom added only if it
  would cover data.
- `ptools.samp_name(s_pts, m1, delta, ctau)`: signal sample name, or None.
- No subplot grids or ratio panels unless requested. If unavoidable, keep `FIGSIZE` and put
  `add_cms_label` on the main axes.
- Older `plot_*` functions:
  - fig/ax passed into `ptools.plot_signal_1D` (style_dict): create with `ptools.new_fig(year)`
    and set `'doCMSLabel': False`, else every call adds another label.
  - `plot_signal_2D`, `plot_bkg_*`, `plot_signal_efficiency` have no such switch yet: use
    `new_fig(year, cms_label=False)` and let them draw the label.
- No meaningful year (BDT plots, mixed-era signal): `year=None`.
- Lumi: don't pass `lumi=` by default. For now only the yield plots in `cutflow_plot_sig_bkg.py`
  show it; check with maintainers before adding it elsewhere. Without lumi, `add_cms_label`
  writes "2024 (13.6 TeV)" itself (mplhep 1.1.2 leaves a stray leading comma).

### 2. Legends

- Always `loc='upper right'`, never `'best'`.
- Order: `plt.tight_layout()` → `legend_headroom` (or `place_legend`) → `plt.tight_layout()`.
- Many/long entries: shrink fontsize (18–20) and/or `ncol=2`; don't move the legend.
- Exception, fixed y-range or meaningless headroom (e.g. efficiency capped at 1, like
  `plot_signal_efficiency` cutflows): try corners upper right → lower left → upper left → lower
  right, take the first clear one per `ptools.legend_covers_data`, else fall back to
  `place_legend`. Never hard-code a corner assuming it's empty.

### 3. Finishing helper

Each script has one (`_finish` / `_finalize_plot`): sets xlabel, ylabel, title, log scales,
legend, then saves and closes.

### 4. 1D distributions (step lines)

- Distributions are step lines: `hep.histplot(h, histtype='step', linewidth=2, yerr=...)`.
  No markers.
- Error bars are Poisson:
  - unweighted fills: sqrt(N) per bin;
  - weighted fills: sqrt(sum w^2) per bin. The histogram must store variances
    (`hist.storage.Weight()`; `Sumw2()` in ROOT).
  Pass them explicitly: `yerr=np.sqrt(h.variances())`. Never pass `yerr=True` on a weighted
  histogram without Weight storage. On that storage hist/mplhep has no sum-of-weights-squared and
  falls back to sqrt(sum w).
- Unit-area plots: divide contents and errors by the same integral (`density=True` does this).
  
- **No default y-axis label**: choose one per plot, or ask.
- Linestyle (`'-'`, `'--'`) for a second grouping dimension.
- Log y when populations span orders of magnitude; log x when the axis is log-binned (check
  `axis.transform is not None`, or keep an explicit set of log-x keys).

### 4c. Efficiencies (points)

- Efficiencies, acceptances and fractions are points with error bars:
  `ax.errorbar(x, eff, yerr=(eff - lo, hi - eff), xerr=half_bin_width, fmt='o', markersize=6,
  capsize=0, lw=2)`. No connecting lines. A cutflow with categorical x uses no `xerr`.
- Error bars: central Clopper–Pearson 68.3% interval from the raw counts k (pass) and n (total),
  via `ptools.clopper_pearson(k, n, cl=0.683)`. Bins with n = 0 are not drawn.
- y range 0 to 1 (headroom above 1 only for the legend, per section 2). The y label names
  numerator and denominator in the caption, not on the axis.
### 4b. Colors (CMS Petroff palettes only)

- ≤ 6 color-coded series: `ptools.CMS_COLORS_6`; 7–10: `ptools.CMS_COLORS_10`. Never mix them on
  one plot. Never use named/matplotlib colors (red, green, `'k'`, tab10, ...) for series.
- Use `ptools.cms_colors(n)` (n = series on the axes), indexed per series. > 10 series: split the
  plot or move a dimension to linestyle.
- `set_plot_style()` makes the 6-color palette the default cycle, so `'C0'`–`'C5'` are CMS
  colors; `'C6'`+ wrap around, so use `cms_colors(n)` beyond 6.
- Fixed category dicts (e.g. merged-category colors) use the palette sized to the category count.
- Backgrounds: `ptools.bkg_cmap` (10-color). Signals overlaid on them:
  `ptools.SIGNAL_COLORS_WITH_BKG` (the 10-color entries `bkg_cmap` leaves free).
- `plot_signal_1D(..., cmap_idx=i, ncolors=n)`: pass `ncolors` so one palette covers the plot;
  without it, series 6+ fall back to the 10-color palette with a warning.
- Black only for neutral elements: data points, a "Total" line, reference/guide lines.

### 5. 2D histograms

- `hep.hist2dplot(h, ax=ax, norm=LogNorm())`; label the colorbar when normalized.
- Meaningful guide lines: y=x as black `':'` axline; cut boundaries as white `'--'`
  axvline/axhline.

### 6. Titles and labels

- Titles: `'<Object>: <Quantity>'` (e.g. `'Merged Electron: $p_T$'`), readable names from a lookup
  dict (`_VAR_TITLES` / `obj_titles`) falling back to the raw key. Slice info on a second line,
  `'\n(...)'`.
- Labels: LaTeX mathtext, units in parentheses `(GeV)` / `(cm)`, never square brackets.
- Signal points: `$m_1$=0.5 GeV, $\delta$=10%, c$\tau$=10 mm` — lowercase delta, splitting as a
  percent (fractional × 100, 0.1 → 10%), not `$\Delta$=0.1`. Format with `ptools.fmt_delta(delta)`
  (also takes a list → `'[10%, 20%]'`).
- Legend entries may carry fractions (`'(merged, selected, 12.3%)'`). Label-only entries:
  `ax.plot([], [], linestyle='none', label=...)`.

### 7. Output

- Save to `plots/<topic>/[subdir]/` (`os.makedirs(..., exist_ok=True)`) as
  `hist_<tag>_<var>_<cut|slice>[_<variant>].png`.
- `plt.close(fig)` after every save.
- Skip empty hists (sum == 0) with a printed message, not a blank plot.
- Script config (outdir/vers/selection/hists_tag/cut/year) at the top as plain module variables.
