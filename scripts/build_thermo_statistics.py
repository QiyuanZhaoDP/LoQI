"""Build THERMO_STATISTICS.md for the LoQI ThermoGen mirror.

Reads downstream_ft/0515_final/per_property/<prop>.csv (post-LoQI 12-element,
post-star-filter view) and emits a single markdown report covering:

  * overall counts (molecules, properties, cells)
  * per-property tier composition (★ histogram)
  * per-property value stats (mean / std / min / p25 / median / p75 / max)
  * per-property source diversity (top sources contributing rows)
  * scaffold counts and fold sizes (random + scaffold splits)
  * per-property value histograms (PNG) — saved to <root>/distributions/

Usage:
    python scripts/build_thermo_statistics.py
    python scripts/build_thermo_statistics.py --root downstream_ft/0515_final
    python scripts/build_thermo_statistics.py --out  custom_path.md
    python scripts/build_thermo_statistics.py --no-plots
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path


STAR = {
    'tier1': 5, 'tier1+confirmed': 5,
    'tier1+disputed': 4, 'tier2': 4, 'tier2+confirmed': 4,
    'tier2+disputed': 3, 'secondary_tight': 3,
    'secondary_loose': 2,
    'secondary_single': 1,
    'downstream': 0,
}


AXIS_OVERRIDES = {
    'visc_liq_298K_cP':             {'logy': True},
    'kinematic_viscosity_298K_cSt': {'logy': True},
    'Q_10ppmv_mgg':                 {'logy': True},
    'dielectric_298K':              {'logy': True},
}


def plot_distributions(per_prop, dist_dir):
    """Write per-property histograms + a composite grid. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    dist_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-property single PNG ----
    for prop, d in sorted(per_prop.items()):
        vals = d['vals']
        if not vals:
            continue
        n = len(vals)
        m = sum(vals) / n
        s = (sum((v-m)**2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        ov = AXIS_OVERRIDES.get(prop, {})
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(vals, bins=min(60, max(10, n // 50)),
                color='#4a7bb7', edgecolor='white', linewidth=0.5)
        if ov.get('logy'):
            ax.set_yscale('log')
        ax.set_title(f'{prop}  (n={n}, μ={m:.3g}, σ={s:.3g})')
        ax.set_xlabel('value'); ax.set_ylabel('count')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(dist_dir / f'{prop}.png', dpi=110)
        plt.close(fig)

    # ---- composite grid ----
    props_with_vals = [(p, d) for p, d in sorted(per_prop.items()) if d['vals']]
    nprops = len(props_with_vals)
    if nprops == 0:
        return True
    ncols = 6
    nrows = (nprops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*2.3))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax, (prop, d) in zip(axes, props_with_vals):
        vals = d['vals']; n = len(vals)
        m = sum(vals) / n
        sd = (sum((v-m)**2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        ov = AXIS_OVERRIDES.get(prop, {})
        ax.hist(vals, bins=min(40, max(8, n // 40)),
                color='#4a7bb7', edgecolor='white', linewidth=0.4)
        if ov.get('logy'):
            ax.set_yscale('log')
        ax.set_title(f'{prop}\nn={n}  μ={m:.3g}  σ={sd:.3g}', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.4)
    for ax in axes[nprops:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(dist_dir / '_all_distributions.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    return True


def quantile(xs, q):
    if not xs:
        return float('nan')
    s = sorted(xs)
    k = (len(s) - 1) * q
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def read_per_property(root: Path):
    """Yield (prop, rows) where rows is a list of dicts with at minimum
    inchikey/smiles/value/tier/sources keys."""
    pp = root / 'per_property'
    for f in sorted(pp.glob('*.csv')):
        prop = f.stem
        rows = list(csv.DictReader(f.open()))
        yield prop, rows


def split_summary(root: Path):
    """Read splits_summary.csv (n_molecules, n_scaffolds, fold sizes)."""
    p = root / 'splits_summary.csv'
    if not p.exists():
        return {}
    out = {}
    for r in csv.DictReader(p.open()):
        out[r['property']] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='downstream_ft/0515_final',
                    help='Root of the LoQI mirror (default: downstream_ft/0515_final)')
    ap.add_argument('--out', default=None,
                    help='Output markdown path (default: <root>/THERMO_STATISTICS.md)')
    ap.add_argument('--no-plots', action='store_true',
                    help='Skip generating per-property histogram PNGs.')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / 'per_property').is_dir():
        raise SystemExit(f'ERROR: {root}/per_property not found')

    out_path = Path(args.out) if args.out else root / 'THERMO_STATISTICS.md'

    sp = split_summary(root)

    # First pass: collect everything.
    per_prop = {}
    all_ik = set()
    overall_tier = Counter()
    overall_source = Counter()
    total_cells = 0

    for prop, rows in read_per_property(root):
        ik_set = set()
        tier_count = Counter()
        src_count = Counter()
        vals = []
        for r in rows:
            ik_set.add(r['inchikey'])
            t = r.get('tier', '')
            tier_count[t] += 1; overall_tier[t] += 1
            for s in (r.get('sources') or '').split('|'):
                if s:
                    src_count[s] += 1; overall_source[s] += 1
            try:
                vals.append(float(r['value']))
            except (ValueError, KeyError):
                pass
        per_prop[prop] = {
            'n': len(rows),
            'n_mol': len(ik_set),
            'tier': tier_count,
            'src': src_count,
            'vals': vals,
        }
        all_ik |= ik_set
        total_cells += len(rows)

    # Star histogram per row (overall).
    star_hist = Counter()
    for t, n in overall_tier.items():
        star_hist[STAR.get(t, 0)] += n

    # ------------------------------- write --------------------------------
    lines: list[str] = []
    w = lines.append

    w(f'# ThermoGen — Per-Property Statistics')
    w('')
    w(f'**Source:** `{root.relative_to(root.parent.parent) if root.is_absolute() else root}/per_property/*.csv`')
    w(f'**Generated:** by `scripts/build_thermo_statistics.py`')
    w('')
    w(f'**Unique molecules:** {len(all_ik):,}     '
      f'**Properties:** {len(per_prop)}     '
      f'**Total cells:** {total_cells:,}')
    w('')

    # ---- 1. Star histogram ----
    w('## 1. Confidence-star distribution (overall, all cells)')
    w('')
    w('| ★ | Count | Share | Tier labels |')
    w('|---:|---:|---:|---|')
    star_labels = {
        5: 'tier1, tier1+confirmed',
        4: 'tier1+disputed, tier2, tier2+confirmed',
        3: 'tier2+disputed, secondary_tight',
        2: 'secondary_loose',
        1: 'secondary_single',
        0: 'downstream (no upstream tier; ML-bench targets)',
    }
    for s in (5, 4, 3, 2, 1, 0):
        n = star_hist.get(s, 0)
        if n == 0:
            continue
        pct = 100 * n / total_cells
        w(f'| {"★"*s + "☆"*(5-s) if s>0 else "—"} | {n:,} | {pct:.1f}% | {star_labels[s]} |')
    w('')

    # ---- 2. Per-property table (tier composition) ----
    w('## 2. Per-property tier composition')
    w('')
    w('Columns: n = total rows, mol = unique molecules, 5★/4★/3★/2★/1★/dn = '
      'cell counts at each confidence level (`dn` = downstream/ML targets).')
    w('')
    w('| Property | n | mol | 5★ | 4★ | 3★ | 2★ | 1★ | dn |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for prop in sorted(per_prop):
        d = per_prop[prop]
        h = Counter()
        for t, n in d['tier'].items():
            h[STAR.get(t, 0)] += n
        def cell(s):  # noqa: E306
            return f'{h.get(s,0):,}' if h.get(s, 0) else ''
        w(f'| `{prop}` | {d["n"]:,} | {d["n_mol"]:,} | '
          f'{cell(5)} | {cell(4)} | {cell(3)} | {cell(2)} | {cell(1)} | {cell(0)} |')
    w('')

    # ---- 3. Per-property value stats ----
    w('## 3. Per-property value statistics')
    w('')
    w('| Property | n | mean | std | min | p25 | median | p75 | max |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for prop in sorted(per_prop):
        v = per_prop[prop]['vals']
        if not v:
            w(f'| `{prop}` | 0 | — | — | — | — | — | — | — |')
            continue
        m = stats.fmean(v)
        s = stats.pstdev(v) if len(v) > 1 else 0.0
        w(f'| `{prop}` | {len(v):,} | {m:.4g} | {s:.4g} | '
          f'{min(v):.4g} | {quantile(v,.25):.4g} | {quantile(v,.5):.4g} | '
          f'{quantile(v,.75):.4g} | {max(v):.4g} |')
    w('')

    # ---- 4. Per-property top sources ----
    w('## 4. Per-property top sources (rows by upstream provider)')
    w('')
    w('Top-3 sources per property, plus total number of distinct sources.')
    w('')
    w('| Property | n_sources | top sources (rows) |')
    w('|---|---:|---|')
    for prop in sorted(per_prop):
        srcs = per_prop[prop]['src']
        if not srcs:
            w(f'| `{prop}` | 0 | (no sources column — downstream) |')
            continue
        top = ', '.join(f'`{s}`: {n}' for s, n in srcs.most_common(3))
        w(f'| `{prop}` | {len(srcs)} | {top} |')
    w('')

    # ---- 5. Splits ----
    w('## 5. Split sizes (random + scaffold, 5-fold)')
    w('')
    if not sp:
        w('_(splits_summary.csv not found — skipped)_')
    else:
        w('| Property | n_mol | n_scaffolds | random fold sizes | scaffold fold sizes |')
        w('|---|---:|---:|---|---|')
        for prop in sorted(sp):
            r = sp[prop]
            w(f'| `{prop}` | {r["n_molecules"]} | {r["n_scaffolds"]} | '
              f'`{r["random_fold_sizes"]}` | `{r["scaffold_fold_sizes"]}` |')
    w('')

    # ---- 6. Per-property distributions (PNG gallery) ----
    if not args.no_plots:
        dist_dir = root / 'distributions'
        ok = plot_distributions(per_prop, dist_dir)
        if ok:
            rel = dist_dir.name  # relative link from the report
            w('## 6. Per-property value distributions')
            w('')
            w(f'Composite (all 43 properties): [`{rel}/_all_distributions.png`]'
              f'({rel}/_all_distributions.png)')
            w('')
            w(f'![all]({rel}/_all_distributions.png)')
            w('')
            w('### Per-property histograms')
            w('')
            w('Click any property name to open the standalone PNG.')
            w('')
            # Render in a 3-column gallery using markdown tables (each cell = thumb).
            props_with_plots = sorted(p for p, d in per_prop.items() if d['vals'])
            ncol = 3
            w('| | | |')
            w('|---|---|---|')
            for i in range(0, len(props_with_plots), ncol):
                chunk = props_with_plots[i:i+ncol]
                cells = []
                for prop in chunk:
                    cells.append(f'**[{prop}]({rel}/{prop}.png)**<br>'
                                 f'<img src="{rel}/{prop}.png" width="280">')
                while len(cells) < ncol:
                    cells.append('')
                w('| ' + ' | '.join(cells) + ' |')
            w('')

    # ---- 7. Top providers across the whole DB ----
    w('## 7. Top 20 upstream sources across all properties')
    w('')
    w('| Source | Rows |')
    w('|---|---:|')
    for s, n in overall_source.most_common(20):
        w(f'| `{s}` | {n:,} |')
    w('')

    out_path.write_text('\n'.join(lines))
    print(f'Wrote {out_path}')
    print(f'  {len(all_ik):,} molecules × {len(per_prop)} properties = {total_cells:,} cells')
    print(f'  ★ histogram: ' + ', '.join(
        f'{s}★={star_hist.get(s,0):,}' for s in (5,4,3,2,1,0) if star_hist.get(s,0)))


if __name__ == '__main__':
    main()
