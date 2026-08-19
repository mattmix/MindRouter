############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/report.py: Report generator. Renders one or
# more harness run directories (corpus, offline accuracy,
# e2e detection, load/overhead) into report.md plus a
# self-contained report.html with embedded PNG charts and
# a rule-engine recommendations table.
#
# Reads artifact files only — never touches the gateway or
# the database. Every section renders defensively: missing
# artifacts become "No data" notes, malformed JSON becomes
# a collected warning, and each chart is skipped when its
# data is absent.
#
############################################################

"""Report generator for the DLP evaluation harness."""

import base64
import html as _htmllib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # host-side rendering, no display
import matplotlib.pyplot as plt          # noqa: E402  (Agg must be set first)
from matplotlib.ticker import PercentFormatter   # noqa: E402

from .constants import DLP_QUEUE_MAXSIZE, GLINER_DEFAULT_MAX_CHARS

# ---------------------------------------------------------------------------
# Palette (dataviz skill, light mode, validated on the white chart surface;
# aqua/yellow sit below 3:1 there — relief is the report's own data tables)
# ---------------------------------------------------------------------------

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#4a3aa7", "#e34948"]
_GOLD = "#F1B300"                       # U of I accent — chrome only, never a series

# Color follows the entity: a scanner mode keeps its hue on every chart.
_MODE_COLORS = {"off": _MUTED, "regex": _SLOTS[0],
                "gliner": _SLOTS[1], "regex+gliner": _SLOTS[2]}
_MODE_ORDER = ["off", "regex", "gliner", "regex+gliner"]

_SEV_RANK = {"critical": 0, "warn": 1, "info": 2}
_SEV_COLORS = {"critical": "#d03b3b", "warn": "#fab219", "info": "#2a78d6"}
_SEVERITY_LEVELS = ["minor", "moderate", "major"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _num(x: Any) -> Optional[float]:
    """x if it is a real number (bools excluded), else None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return x


def _fmt(x: Any, nd: int = 3) -> str:
    if _num(x) is None:
        return "—"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:.{nd}f}"


def _fmt_int(x: Any) -> str:
    return f"{int(x):,}" if _num(x) is not None else "—"


def _fmt_pct(x: Any, nd: int = 1) -> str:
    return f"{100.0 * x:.{nd}f}%" if _num(x) is not None else "—"


def _fmt_ms(x: Any) -> str:
    v = _num(x)
    if v is None:
        return "—"
    if v >= 100:
        return f"{v:,.0f}"
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def _lat_cells(d: Any) -> List[str]:
    d = d if isinstance(d, dict) else {}
    return ([_fmt_int(d.get("n"))] +
            [_fmt_ms(d.get(k)) for k in ("mean", "p50", "p90", "p95", "p99", "max")])


_LAT_HEADERS = ["", "n", "mean", "p50", "p90", "p95", "p99", "max"]


def _bucket_chars(label: Any) -> Optional[float]:
    """Representative char count for a 'lo-hi' / '>=N' length-bucket label."""
    nums = re.findall(r"\d+", str(label))
    if not nums:
        return None
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2.0
    return float(nums[0])


def _sweep_partial_areas(sweep: dict) -> Tuple[Any, Any]:
    """(pr_area, roc_area) from a sweep dict, accepting both key spellings.

    The metrics side is moving to pr_auc_partial/roc_auc_partial (the numbers
    are trapezoid areas over the swept points only, not full-curve AUCs);
    older artifacts carry pr_auc/roc_auc. Prefer the explicit names.
    """
    pr = sweep.get("pr_auc_partial")
    roc = sweep.get("roc_auc_partial")
    return (pr if _num(pr) is not None else sweep.get("pr_auc"),
            roc if _num(roc) is not None else sweep.get("roc_auc"))


def _obs_span(vals: List[Any]) -> str:
    """'[lo, hi]' over the defined values — the observed range of a partial curve."""
    vs = [v for v in (_num(v) for v in vals) if v is not None]
    return f"[{min(vs):.3f}, {max(vs):.3f}]" if vs else "[—]"


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_json(path: str, warnings: List[str]) -> Any:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        warnings.append(f"malformed JSON skipped: {path}: {e}")
        return None


def _count_jsonl(path: str) -> Optional[int]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def load_run_data(run_dirs: List[str]) -> Dict[str, Any]:
    """Merge the artifacts of one or more run directories into one dict.

    Offline / e2e / corpus / config artifacts: first directory that has one
    wins (extras are noted). Load artifacts: phases and baseline rows from
    every directory are concatenated so a matrix split across runs charts
    as one series set.
    """
    data: Dict[str, Any] = {
        "runs": [], "warnings": [],
        "corpus_manifest": None, "corpus_dir": None,
        "offline": None, "offline_dir": None, "offline_findings_count": None,
        "e2e_metrics": None, "e2e_dir": None, "e2e_results_count": None,
        "load_phases": [], "baseline_comparison": [], "load_dirs": [],
        "load_requests_count": 0, "cpu_samples_count": 0,
        "config_snapshot": None, "config_dir": None,
    }
    warnings = data["warnings"]
    for rd in run_dirs:
        rd = os.path.abspath(rd)
        if not os.path.isdir(rd):
            warnings.append(f"run directory not found: {rd}")
            continue
        manifest = _load_json(os.path.join(rd, "run.json"), warnings)
        data["runs"].append({"dir": rd,
                             "manifest": manifest if isinstance(manifest, dict) else None})

        cm = _load_json(os.path.join(rd, "manifest.json"), warnings)
        if isinstance(cm, dict) and data["corpus_manifest"] is None:
            data["corpus_manifest"], data["corpus_dir"] = cm, rd

        om = _load_json(os.path.join(rd, "offline_metrics.json"), warnings)
        if isinstance(om, dict):
            if data["offline"] is None:
                data["offline"], data["offline_dir"] = om, rd
                data["offline_findings_count"] = _count_jsonl(
                    os.path.join(rd, "offline_findings.jsonl"))
            else:
                warnings.append(f"extra offline_metrics.json ignored: {rd}")

        em = _load_json(os.path.join(rd, "e2e_metrics.json"), warnings)
        if isinstance(em, dict):
            if data["e2e_metrics"] is None:
                data["e2e_metrics"], data["e2e_dir"] = em, rd
                data["e2e_results_count"] = _count_jsonl(
                    os.path.join(rd, "e2e_results.jsonl"))
            else:
                warnings.append(f"extra e2e_metrics.json ignored: {rd}")

        lp = _load_json(os.path.join(rd, "load_phases.json"), warnings)
        if isinstance(lp, dict):
            data["load_dirs"].append(rd)
            phases = lp.get("phases")
            if isinstance(phases, list):
                data["load_phases"].extend(p for p in phases if isinstance(p, dict))
            bc = lp.get("baseline_comparison")
            if isinstance(bc, list):
                data["baseline_comparison"].extend(r for r in bc if isinstance(r, dict))
            data["load_requests_count"] += _count_jsonl(
                os.path.join(rd, "load_requests.jsonl")) or 0
            data["cpu_samples_count"] += _count_jsonl(
                os.path.join(rd, "cpu_samples.jsonl")) or 0

        cs = _load_json(os.path.join(rd, "config_snapshot.json"), warnings)
        if isinstance(cs, dict) and data["config_snapshot"] is None:
            data["config_snapshot"], data["config_dir"] = cs, rd
    return data


# ---------------------------------------------------------------------------
# Recommendations rule engine
# ---------------------------------------------------------------------------

def build_recommendations(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Rule engine over the merged run data.

    Returns [{"severity": "info"|"warn"|"critical", "finding", "recommendation"}]
    sorted most-severe first. Silent (empty list) on healthy inputs; every
    firing rule cites the numbers that triggered it.
    """
    recs: List[Dict[str, str]] = []
    off = data.get("offline") if isinstance(data.get("offline"), dict) else {}
    e2e = data.get("e2e_metrics") if isinstance(data.get("e2e_metrics"), dict) else {}
    phases = [p for p in (data.get("load_phases") or []) if isinstance(p, dict)]
    baseline = [r for r in (data.get("baseline_comparison") or []) if isinstance(r, dict)]

    def add(severity: str, finding: str, recommendation: str) -> None:
        recs.append({"severity": severity, "finding": finding,
                     "recommendation": recommendation})

    # -- credit_card span precision -> Luhn post-validation ------------------
    cc = ((off.get("span_confusion") or {}).get("per_category") or {}).get("credit_card") or {}
    ccp = _num(cc.get("precision"))
    if ccp is not None and ccp < 0.80:
        add("warn",
            f"credit_card span precision is {ccp:.2f} "
            f"(tp={_fmt_int(cc.get('tp'))}, fp={_fmt_int(cc.get('fp'))}), "
            f"below the 0.80 floor.",
            "Add Luhn post-validation: a custom dlp.regex.patterns entry (or a "
            "validation hook) that drops 13–19 digit candidates failing the Luhn "
            "check before an alert is raised.")

    # -- document specificity -> FP flood risk -------------------------------
    dc = off.get("doc_confusion") or {}
    spec = _num(dc.get("specificity"))
    if spec is not None and spec < 0.95:
        traps = off.get("fp_traps") or {}
        top = sorted(((k, v) for k, v in traps.items() if _num(v) is not None),
                     key=lambda kv: -kv[1])[:3]
        trap_txt = (", ".join(f"{k} ({_fmt_int(v)})" for k, v in top)
                    or "no trap attribution recorded")
        add("warn",
            f"Document specificity is {spec:.3f} (< 0.95; fp={_fmt_int(dc.get('fp'))}, "
            f"tn={_fmt_int(dc.get('tn'))}) — clean traffic will flood reviewers with "
            f"false alerts. Top FP traps: {trap_txt}.",
            "Tune the patterns/labels behind the named traps (tighter boundaries or "
            "an allowlist for the lookalike formats) before widening DLP rollout.")

    # -- threshold sweep vs the 0.5 default ----------------------------------
    best = (off.get("sweep") or {}).get("best_f1") or {}
    bt = _num(best.get("threshold"))
    if bt is not None and abs(bt - 0.5) > 0.05:
        f1_txt = _fmt(best.get("f1"), 3)
        add("info",
            f"Threshold sweep peaks at t={bt:.2f} (span F1={f1_txt}), "
            f"{bt - 0.5:+.2f} from the current 0.50 default.",
            f"Set dlp.gliner.threshold to {bt:.2f} (Admin → DLP) to operate at the "
            f"sweep's best span-F1 point.")

    # -- load coverage: scans dropped under load -----------------------------
    dropped = []
    for ph in phases:
        if str(ph.get("scanner_mode")) == "off":
            continue    # DLP disabled by design: zero alerts is not scan loss
        dlp = ph.get("dlp") or {}
        cov = _num(dlp.get("coverage_rate"))
        if cov is not None and cov < 0.999:
            dropped.append((str(ph.get("phase_id")), cov, dlp.get("queue_drops_logged")))
    if dropped:
        detail = "; ".join(
            f"{pid}: coverage {cov:.3f}"
            + (f", {_fmt_int(drops)} 'dlp_queue_full' drops logged"
               if _num(drops) is not None else ", queue-drop count unavailable")
            for pid, cov, drops in dropped)
        add("critical",
            f"Scans were dropped under load — coverage below 0.999 in "
            f"{len(dropped)} phase(s): {detail}.",
            f"The per-worker asyncio queue (maxsize {DLP_QUEUE_MAXSIZE:,}) drops "
            f"silently when full. Export a queue-depth metric and move the DLP "
            f"worker to a multi-consumer pool so scan throughput tracks offered load.")

    # -- drain time: scan backlog risk ---------------------------------------
    slow = [(str(ph.get("phase_id")),
             _num((ph.get("dlp") or {}).get("drain_seconds")))
            for ph in phases]
    slow = [(pid, d) for pid, d in slow if d is not None and d > 60]
    e2e_drain = _num((e2e.get("drain") or {}).get("seconds"))
    if e2e_drain is not None and e2e_drain > 60:
        slow.append(("e2e", e2e_drain))
    if slow:
        detail = "; ".join(f"{pid}: {d:.0f}s" for pid, d in slow)
        add("warn",
            f"Scan backlog risk — queue drain exceeded 60s after send stopped "
            f"({detail}).",
            "Alerts lag real traffic by the drain time. Increase scan throughput "
            "(more consumers, lower per-scan latency) before enabling heavier "
            "scanners in production.")

    # -- DLP overhead vs baseline --------------------------------------------
    heavy = []
    for row in baseline:
        p95 = _num(row.get("e2e_p95_delta_ms"))
        thr = _num(row.get("throughput_delta_pct"))
        if (p95 is not None and p95 > 250) or (thr is not None and thr < -10):
            heavy.append(f"{row.get('mode')}@c{row.get('concurrency')}: "
                         f"p95 {_fmt_ms(p95)}ms, throughput "
                         f"{_fmt(thr, 1)}% vs off")
    if heavy:
        add("warn",
            "DLP overhead exceeds budget (p95 delta > 250ms or throughput "
            "delta < -10%) — " + "; ".join(heavy) + ".",
            "DLP is designed to be post-hoc with zero user latency; a delta this "
            "size means scanning is contending with the request path (CPU or DB). "
            "Profile the worker and isolate its resources.")

    # -- degraded scans (scanner errors) -------------------------------------
    sources = []
    off_err = _num(off.get("scan_errors"))
    if off_err:
        sources.append(f"offline scan_errors={_fmt_int(off_err)}")
    e2e_err = _num(e2e.get("scanner_error_alerts"))
    if e2e_err:
        sources.append(f"e2e scanner_error_alerts={_fmt_int(e2e_err)}")
    load_err = sum(int(_num((ph.get("dlp") or {}).get("scanner_error_alerts")) or 0)
                   for ph in phases)
    if load_err:
        sources.append(f"load scanner_error_alerts={load_err:,}")
    if sources:
        add("warn",
            "Degraded scans detected: " + ", ".join(sources) + ".",
            "Each error is a document that got no real scan (synthetic "
            "'dlp_scanner_error' alert rows, excluded from accuracy). Inspect the "
            "scanner exception log before trusting the coverage numbers.")

    # -- truncation blindness past the GLiNER scan cap -----------------------
    rb = off.get("recall_by") if isinstance(off.get("recall_by"), dict) else {}
    depth = rb.get("depth") or off.get("recall_by_depth")   # nested (producer) or flat (legacy)
    if isinstance(depth, dict):
        deep_bad, shallow_ok = [], []
        for bucket, row in depth.items():
            row = row if isinstance(row, dict) else {}
            r, n = _num(row.get("recall")), _num(row.get("n"))
            m = re.search(r"\d+", str(bucket))
            if r is None or not n or m is None:
                continue
            pos = int(m.group())
            if pos >= GLINER_DEFAULT_MAX_CHARS and r <= 0.5:
                deep_bad.append((str(bucket), r, int(n)))
            elif pos < GLINER_DEFAULT_MAX_CHARS and r >= 0.8:
                shallow_ok.append((str(bucket), r))
        if deep_bad and shallow_ok:
            deep_txt = ", ".join(f"{b} recall {r:.2f} (n={n})" for b, r, n in deep_bad)
            sh_b, sh_r = shallow_ok[0]
            add("info",
                f"Truncation blindness: entities past the GLiNER "
                f"{GLINER_DEFAULT_MAX_CHARS:,}-char scan cap go undetected — "
                f"{deep_txt}, vs {sh_r:.2f} at depth {sh_b}.",
                "Raise dlp.gliner.max_scan_chars or chunk long documents for GLiNER; "
                "regex still covers the full 200,000-char window, so only "
                "GLiNER-exclusive categories are blind at depth.")

    recs.sort(key=lambda r: _SEV_RANK.get(r["severity"], 3))
    return recs


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style_axes(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _new_fig(w: float = 7.2, h: float = 4.2, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), dpi=140)
    fig.patch.set_facecolor(_SURFACE)
    for ax in (axes if ncols > 1 else [axes]):
        _style_axes(ax)
    return fig, axes


def _titles(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, color=_INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=_INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=_INK_2, fontsize=9)


def _legend(ax) -> None:
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_2)


def _line(ax, xs, ys, color, label=None):
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
            markerfacecolor=color, markeredgecolor=_SURFACE, markeredgewidth=1.5,
            solid_capstyle="round", solid_joinstyle="round", label=label)


def _phase_series(phases: List[dict],
                  getter: Callable[[dict], Any]) -> Dict[str, List[Tuple[float, float]]]:
    """{mode: sorted [(concurrency, mean-of-values)]} from load phases."""
    by_mode: Dict[str, Dict[float, List[float]]] = {}
    for ph in phases or []:
        mode = str(ph.get("scanner_mode") or "?")
        conc, y = _num(ph.get("concurrency")), _num(getter(ph))
        if conc is None or y is None:
            continue
        by_mode.setdefault(mode, {}).setdefault(conc, []).append(y)
    return {m: sorted((c, sum(v) / len(v)) for c, v in pts.items())
            for m, pts in by_mode.items() if pts}


def _plot_mode_lines(ax, series: Dict[str, List[Tuple[float, float]]]) -> None:
    modes = ([m for m in _MODE_ORDER if m in series] +
             sorted(m for m in series if m not in _MODE_COLORS))
    extra = [c for c in _SLOTS if c not in _MODE_COLORS.values()]
    for i, mode in enumerate(modes):
        color = _MODE_COLORS.get(mode) or extra[min(i, len(extra) - 1)]
        xs, ys = zip(*series[mode])
        _line(ax, xs, ys, color, label=mode)
    if len(modes) >= 2:
        _legend(ax)


def _chart_per_category_recall(data: dict):
    off = data.get("offline") or {}
    lenient = (off.get("span_confusion") or {}).get("per_category") or {}
    strict = (off.get("span_confusion_strict") or {}).get("per_category") or {}
    cats = [c for c in dict.fromkeys(list(lenient) + list(strict))
            if _num((lenient.get(c) or {}).get("recall")) is not None
            or _num((strict.get(c) or {}).get("recall")) is not None]
    if not cats:
        return None
    lv = [_num((lenient.get(c) or {}).get("recall")) or 0.0 for c in cats]
    sv = [_num((strict.get(c) or {}).get("recall")) or 0.0 for c in cats]
    fig, ax = _new_fig(max(7.2, 1.1 * len(cats)), 4.2)
    xs = range(len(cats))
    ax.bar([x - 0.19 for x in xs], lv, width=0.36, color=_SLOTS[0],
           linewidth=0, label="lenient")
    ax.bar([x + 0.19 for x in xs], sv, width=0.36, color=_SLOTS[1],
           linewidth=0, label="strict")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _titles(ax, "Per-category span recall — lenient vs strict", ylabel="recall")
    _legend(ax)
    return fig, "Per-category span recall (lenient: any overlapping finding; strict: category must also match)."


def _chart_fp_traps(data: dict):
    traps = (data.get("offline") or {}).get("fp_traps") or {}
    items = sorted(((k, v) for k, v in traps.items() if _num(v) and v > 0),
                   key=lambda kv: -kv[1])
    total = sum(v for _, v in items)
    if not items or not total:
        return None
    names = [k for k, _ in items]
    shares = [100.0 * v / total for _, v in items]
    cum, running = [], 0.0
    for s in shares:
        running += s
        cum.append(running)
    fig, ax = _new_fig(max(7.2, 0.9 * len(names)), 4.2)
    ax.bar(range(len(names)), shares, width=0.6, color=_SLOTS[0],
           linewidth=0, label="share of FPs")
    _line(ax, range(len(names)), cum, _SLOTS[1], label="cumulative")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    _titles(ax, "False-positive traps — Pareto", ylabel="share of false positives")
    _legend(ax)
    return fig, f"False-positive attribution by trap ({total:,} trap-attributed FPs)."


def _chart_sweep(data: dict):
    sweep = (data.get("offline") or {}).get("sweep") or {}
    pts = [p for p in (sweep.get("points") or []) if isinstance(p, dict)]
    pr = [(p.get("span_recall"), p.get("span_precision"), p.get("threshold"))
          for p in pts]
    pr = [(x, y, t) for x, y, t in pr if _num(x) is not None and _num(y) is not None]
    roc = [(p.get("doc_fpr"), p.get("doc_recall")) for p in pts]
    roc = [(x, y) for x, y in roc if _num(x) is not None and _num(y) is not None]
    if not pr and not roc:
        return None
    fig, axes = _new_fig(9.6, 4.2, ncols=2)
    ax_pr, ax_roc = axes
    if pr:
        pr_sorted = sorted(pr)
        _line(ax_pr, [p[0] for p in pr_sorted], [p[1] for p in pr_sorted], _SLOTS[0])
        best_t = _num((sweep.get("best_f1") or {}).get("threshold"))
        if best_t is not None:
            hit = [p for p in pr if p[2] is not None and abs(p[2] - best_t) < 1e-9]
            if hit:
                x, y, t = hit[0]
                ax_pr.plot([x], [y], marker="o", markersize=9, color=_SLOTS[1],
                           markeredgecolor=_SURFACE, markeredgewidth=1.5, zorder=5)
                ax_pr.annotate(f"t={t:.2f}", (x, y), textcoords="offset points",
                               xytext=(8, -4), fontsize=9, color=_INK_2)
    _titles(ax_pr, "Precision–recall (span level)", "recall", "precision")
    ax_pr.set_xlim(0, 1.02)
    ax_pr.set_ylim(0, 1.05)
    if roc:
        # No 0.5 chance diagonal: the curve covers only the swept thresholds,
        # so a full-ROC chance reference would invite a bogus comparison.
        roc_sorted = sorted(roc)
        _line(ax_roc, [p[0] for p in roc_sorted], [p[1] for p in roc_sorted], _SLOTS[0])
    _titles(ax_roc, "ROC (document level)", "false-positive rate", "recall")
    ax_roc.set_xlim(0, 1.02)
    ax_roc.set_ylim(0, 1.05)
    fig.tight_layout()
    pr_area, roc_area = _sweep_partial_areas(sweep)
    cap = (f"Confidence-threshold sweep. Partial PR area {_fmt(pr_area, 3)} over "
           f"observed recall {_obs_span([p[0] for p in pr])}; partial ROC area "
           f"{_fmt(roc_area, 3)} over observed FPR {_obs_span([p[0] for p in roc])} "
           f"(trapezoid over swept points only — not full-curve AUCs).")
    return fig, cap


def _chart_latency_by_length(data: dict):
    raw = (data.get("offline") or {}).get("latency_by_length")
    if isinstance(raw, dict):
        # producer shape (offline_eval): {"lo-hi" | ">=N": {"n", "regex", "gliner"}}
        rows = [{"chars": _bucket_chars(label),
                 "regex_ms": row.get("regex"), "gliner_ms": row.get("gliner")}
                for label, row in raw.items() if isinstance(row, dict)]
    else:                                        # legacy list of {chars, *_ms} rows
        rows = [r for r in (raw or []) if isinstance(r, dict)]
    rows = [r for r in rows if _num(r.get("chars")) is not None]
    rx = sorted((r["chars"], r["regex_ms"]) for r in rows if _num(r.get("regex_ms")) is not None)
    gl = sorted((r["chars"], r["gliner_ms"]) for r in rows if _num(r.get("gliner_ms")) is not None)
    if not rx and not gl:
        return None
    fig, ax = _new_fig()
    if rx:
        _line(ax, [p[0] for p in rx], [p[1] for p in rx], _SLOTS[0], label="regex")
    if gl:
        _line(ax, [p[0] for p in gl], [p[1] for p in gl], _SLOTS[1], label="gliner")
    chars = [p[0] for p in rx + gl]
    if chars and min(chars) > 0 and max(chars) / min(chars) > 50:
        ax.set_xscale("log")
    _titles(ax, "Scan latency vs document length", "document chars", "scan latency (ms)")
    if rx and gl:
        _legend(ax)
    return fig, "Per-scanner scan latency as document length grows."


def _chart_load_e2e(data: dict):
    phases = data.get("load_phases") or []
    s50 = _phase_series(phases, lambda p: ((p.get("latency_ms") or {}).get("e2e") or {}).get("p50"))
    s95 = _phase_series(phases, lambda p: ((p.get("latency_ms") or {}).get("e2e") or {}).get("p95"))
    if not s50 and not s95:
        return None
    fig, axes = _new_fig(9.6, 4.2, ncols=2)
    _plot_mode_lines(axes[0], s50)
    _titles(axes[0], "E2E latency p50 vs concurrency", "concurrency", "latency (ms)")
    _plot_mode_lines(axes[1], s95)
    _titles(axes[1], "E2E latency p95 vs concurrency", "concurrency", "latency (ms)")
    fig.tight_layout()
    return fig, "Request end-to-end latency by scanner mode as concurrency grows."


def _mode_line_chart(data: dict, getter, title: str, ylabel: str,
                     caption: str, percent: bool = False):
    series = _phase_series(data.get("load_phases") or [], getter)
    if not series:
        return None
    fig, ax = _new_fig()
    _plot_mode_lines(ax, series)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(top=max(1.01, ax.get_ylim()[1]))
    _titles(ax, title, "concurrency", ylabel)
    return fig, caption


_CHART_BUILDERS: List[Tuple[str, Callable[[dict], Any]]] = [
    ("per_category_recall", _chart_per_category_recall),
    ("fp_trap_pareto", _chart_fp_traps),
    ("threshold_sweep", _chart_sweep),
    ("scan_latency_vs_length", _chart_latency_by_length),
    ("load_e2e_latency", _chart_load_e2e),
    ("load_throughput", lambda d: _mode_line_chart(
        d, lambda p: (p.get("offered") or {}).get("rps"),
        "Throughput vs concurrency", "requests / s",
        "Achieved request throughput by scanner mode.")),
    ("load_coverage", lambda d: _mode_line_chart(
        d, lambda p: (p.get("dlp") or {}).get("coverage_rate"),
        "DLP coverage vs concurrency", "coverage",
        "Share of dirty documents that produced an alert.", percent=True)),
    ("load_scan_lag", lambda d: _mode_line_chart(
        d, lambda p: ((p.get("dlp") or {}).get("scan_lag_ms") or {}).get("p95"),
        "Scan lag p95 vs concurrency", "lag (ms)",
        "p95 delay between request completion and its DLP scan.")),
    ("load_drain", lambda d: _mode_line_chart(
        d, lambda p: (p.get("dlp") or {}).get("drain_seconds"),
        "Queue drain vs concurrency", "drain (s)",
        "Seconds for the scan queue to settle after send stopped.")),
    ("load_cpu", lambda d: _mode_line_chart(
        d, lambda p: (p.get("cpu") or {}).get("app_mean_pct"),
        "App CPU vs concurrency", "CPU (%)",
        "Mean app-container CPU by scanner mode.")),
]


def _build_charts(data: dict, out_dir: str) -> Dict[str, Dict[str, str]]:
    charts: Dict[str, Dict[str, str]] = {}
    charts_dir = os.path.join(out_dir, "charts")
    for name, builder in _CHART_BUILDERS:
        try:
            result = builder(data)
        except Exception as e:                       # a bad chart never kills the report
            data["warnings"].append(f"chart {name} skipped: {e!r}")
            continue
        if result is None:
            continue
        fig, caption = result
        os.makedirs(charts_dir, exist_ok=True)
        path = os.path.join(charts_dir, f"{name}.png")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        charts[name] = {"path": path, "file": f"charts/{name}.png",
                        "b64": b64, "caption": caption}
    return charts


# ---------------------------------------------------------------------------
# Section model (shared by the markdown and HTML renderers)
#
# Block kinds: ("h3", text) ("p", text) ("note", text) ("pre", text)
#              ("table", headers, rows[, cls]) ("chart", name)
#              ("tiles", [(label, value, sub)]) ("recs", recs)
# ---------------------------------------------------------------------------

_NO_DATA = "No data — "


def _kv_rows(d: dict, fmt_val=lambda v: v if isinstance(v, str) else json.dumps(v)) -> List[List[str]]:
    return [[str(k), str(fmt_val(v))] for k, v in d.items()]


def _count_table(d: Any, key_header: str, sort_desc: bool = True):
    if not isinstance(d, dict) or not d:
        return None
    items = list(d.items())
    if sort_desc:
        items.sort(key=lambda kv: -(_num(kv[1]) or 0))
    return ("table", [key_header, "count"],
            [[str(k), _fmt_int(v)] for k, v in items])


def _recall_table(d: Any, key_header: str):
    if not isinstance(d, dict) or not d:
        return None
    rows = [[str(g), _fmt_int((r or {}).get("tp")), _fmt_int((r or {}).get("fn")),
             _fmt_int((r or {}).get("n")), _fmt_pct((r or {}).get("recall"))]
            for g, r in sorted(d.items())]
    return ("table", [key_header, "tp", "fn", "n", "recall"], rows)


def _severity_blocks(sev: Any, heading: str) -> List[tuple]:
    if not isinstance(sev, dict) or not isinstance(sev.get("matrix"), dict):
        return []
    matrix = sev["matrix"]
    levels = ([l for l in _SEVERITY_LEVELS if l in matrix
               or any(l in (r or {}) for r in matrix.values())] +
              sorted(k for k in matrix if k not in _SEVERITY_LEVELS))
    rows = [[f"expected {e}"] + [_fmt_int((matrix.get(e) or {}).get(p, 0)) for p in levels]
            for e in levels if e in matrix]
    blocks: List[tuple] = [("h3", heading),
                           ("table", [""] + [f"predicted {p}" for p in levels], rows)]
    emr = _num(sev.get("exact_match_rate"))
    if emr is not None:
        blocks.append(("p", f"Exact severity match rate: {_fmt_pct(emr)}"
                            + (f" over {_fmt_int(sev.get('n'))} alerting docs."
                               if _num(sev.get("n")) is not None else ".")))
    return blocks


def _sec_executive(data, recs, charts) -> dict:
    blocks: List[tuple] = []
    off = data.get("offline") or {}
    dc = off.get("doc_confusion") or {}
    span = (off.get("span_confusion") or {}).get("overall") or {}
    e2e = data.get("e2e_metrics") or {}
    cov = e2e.get("coverage") or {}
    tiles = []
    if _num(dc.get("recall")) is not None:
        tiles.append(("Offline doc recall", _fmt_pct(dc["recall"]), "dirty docs flagged"))
    if _num(dc.get("specificity")) is not None:
        tiles.append(("Offline specificity", _fmt_pct(dc["specificity"]), "clean docs passed"))
    if _num(span.get("f1")) is not None:
        tiles.append(("Span F1 (lenient)", _fmt(span["f1"], 3), "entity-level, all scanners"))
    rate = _num(cov.get("in_scope_rate"))
    if rate is not None:
        tiles.append(("E2E in-scope coverage", _fmt_pct(rate), "alerts on in-scope dirty docs"))
    elif _num(cov.get("rate")) is not None:
        tiles.append(("E2E coverage", _fmt_pct(cov["rate"]), "alerts on dirty docs"))
    if _num((e2e.get("clean_fp") or {}).get("rate")) is not None:
        tiles.append(("E2E clean FP rate", _fmt_pct(e2e["clean_fp"]["rate"]),
                      "alerts on clean docs"))
    load_cov = [c for c in (_num((p.get("dlp") or {}).get("coverage_rate"))
                            for p in data.get("load_phases") or []
                            if str(p.get("scanner_mode")) != "off") if c is not None]
    if load_cov:
        tiles.append(("Worst load coverage", _fmt_pct(min(load_cov)), "across load phases"))
    p95s = [d for d in (_num(r.get("e2e_p95_delta_ms"))
                        for r in data.get("baseline_comparison") or []) if d is not None]
    if p95s:
        tiles.append(("Max p95 overhead", f"{_fmt_ms(max(p95s))} ms", "vs scanners-off baseline"))
    if tiles:
        blocks.append(("tiles", tiles))
    else:
        blocks.append(("note", _NO_DATA + "no metric artifacts found in the supplied run directories."))
    blocks.append(("h3", "Recommendations"))
    if recs:
        blocks.append(("recs", recs))
    else:
        blocks.append(("p", "No findings — every rule-engine check passed on the available data."))
    if data.get("warnings"):
        blocks.append(("h3", "Warnings"))
        blocks.append(("table", ["warning"], [[w] for w in data["warnings"]], "text"))
    return {"id": "executive-summary", "title": "Executive summary", "blocks": blocks}


def _sec_corpus(data) -> dict:
    blocks: List[tuple] = []
    cm = data.get("corpus_manifest")
    if not isinstance(cm, dict):
        blocks.append(("note", _NO_DATA + "no corpus manifest.json in the supplied run directories."))
    else:
        blocks.append(("table", ["", "count"],
                       [[k, _fmt_int(cm.get(k))] for k in
                        ("docs", "dirty_docs", "clean_docs", "entities", "negatives")]))
        for key, header, sort_desc in (
                ("by_profile", "profile", True), ("by_carrier", "carrier", True),
                ("by_category", "category", True), ("by_difficulty", "difficulty", True),
                ("char_histogram", "doc length (chars)", False)):
            t = _count_table(cm.get(key), header, sort_desc)
            if t:
                blocks.append(("h3", header.capitalize() if key != "char_histogram"
                               else "Document length histogram"))
                blocks.append(t)
    return {"id": "corpus", "title": "Corpus", "blocks": blocks}


def _sec_offline(data, charts) -> dict:
    blocks: List[tuple] = []
    off = data.get("offline")
    if not isinstance(off, dict):
        blocks.append(("note", _NO_DATA + "no offline_metrics.json in the supplied run directories."))
        return {"id": "offline-accuracy", "title": "Offline accuracy", "blocks": blocks}

    cfg = off.get("run") if isinstance(off.get("run"), dict) else off.get("scanner_config")
    if isinstance(cfg, dict) and cfg:
        blocks.append(("h3", "Scanner configuration"))
        blocks.append(("pre", json.dumps(cfg, indent=2, sort_keys=True)))

    dc = off.get("doc_confusion")
    if isinstance(dc, dict):
        blocks.append(("h3", "Document-level confusion"))
        blocks.append(("table", ["tp", "fp", "tn", "fn"],
                       [[_fmt_int(dc.get(k)) for k in ("tp", "fp", "tn", "fn")]]))
        blocks.append(("table", ["precision", "recall", "specificity", "f1", "mcc", "fpr", "fnr"],
                       [[_fmt(dc.get("precision")), _fmt(dc.get("recall")),
                         _fmt(dc.get("specificity")), _fmt(dc.get("f1")),
                         _fmt(dc.get("mcc")), _fmt(dc.get("fpr")), _fmt(dc.get("fnr"))]]))

    lenient = off.get("span_confusion") or {}
    strict = off.get("span_confusion_strict") or {}
    if isinstance(lenient.get("overall"), dict) or isinstance(strict.get("overall"), dict):
        blocks.append(("h3", "Span-level confusion (overall)"))
        rows = []
        for label, d in (("lenient", lenient.get("overall")), ("strict", strict.get("overall"))):
            if isinstance(d, dict):
                rows.append([label, _fmt_int(d.get("tp")), _fmt_int(d.get("fn")),
                             _fmt_int(d.get("fp")), _fmt(d.get("precision")),
                             _fmt(d.get("recall")), _fmt(d.get("f1"))])
        blocks.append(("table", ["matching", "tp", "fn", "fp", "precision", "recall", "f1"], rows))

    pc_l, pc_s = lenient.get("per_category") or {}, strict.get("per_category") or {}
    cats = list(dict.fromkeys(list(pc_l) + list(pc_s)))
    if cats:
        blocks.append(("h3", "Per-category span metrics"))
        rows = []
        for c in cats:
            a, b = pc_l.get(c) or {}, pc_s.get(c) or {}
            rows.append([c, _fmt_int(a.get("tp")), _fmt_int(a.get("fn")), _fmt_int(a.get("fp")),
                         _fmt(a.get("precision")), _fmt(a.get("recall")), _fmt(a.get("f1")),
                         _fmt(b.get("recall")), _fmt(b.get("f1"))])
        blocks.append(("table",
                       ["category", "tp", "fn", "fp", "precision", "recall", "f1",
                        "strict recall", "strict f1"], rows))
        blocks.append(("chart", "per_category_recall"))

    boot = off.get("bootstrap")
    if isinstance(boot, dict) and boot:
        rows = []
        for keys, label in ((("doc_recall", "doc_recall_ci"), "doc recall"),
                            (("doc_precision", "doc_precision_ci"), "doc precision"),
                            (("span_recall", "span_recall_ci"), "span recall")):
            ci = next((boot[k] for k in keys if isinstance(boot.get(k), dict)), None)
            if ci is not None:
                rows.append([label, _fmt(ci.get("point")), _fmt(ci.get("lo")), _fmt(ci.get("hi"))])
        if rows:                                 # heading only when there is a table
            blocks.append(("h3", "Bootstrap 95% confidence intervals"))
            blocks.append(("table", ["metric", "point", "CI lo", "CI hi"], rows))

    t = _recall_table(off.get("scope_split"), "scope")
    if t:
        blocks.append(("h3", "Scope split (in-scope vs out-of-scope recall)"))
        blocks.append(t)
    rb = off.get("recall_by") if isinstance(off.get("recall_by"), dict) else {}
    for header in ("difficulty", "generator", "carrier", "depth"):
        t = _recall_table(rb.get(header) or off.get(f"recall_by_{header}"), header)
        if t:
            blocks.append(("h3", f"Recall by {header}"))
            blocks.append(t)

    blocks.extend(_severity_blocks(off.get("severity_accuracy") or off.get("severity"),
                                   "Severity confusion"))

    traps = off.get("fp_traps")
    t = _count_table(traps, "trap")
    if t:
        blocks.append(("h3", "False-positive traps"))
        blocks.append(t)
        blocks.append(("chart", "fp_trap_pareto"))

    lat_ms = off.get("latency_ms") if isinstance(off.get("latency_ms"), dict) else {}
    lat = lat_ms.get("per_scanner")
    if not isinstance(lat, dict):
        lat = off.get("latency")                 # legacy flat shape
    if isinstance(lat, dict):
        rows = [[name] + _lat_cells(lat.get(name)) for name in ("regex", "gliner")
                if isinstance(lat.get(name), dict)]
        if isinstance(lat_ms.get("per_doc_total"), dict):
            rows.append(["per-doc total"] + _lat_cells(lat_ms["per_doc_total"]))
        if rows:
            blocks.append(("h3", "Scan latency (ms)"))
            blocks.append(("table", _LAT_HEADERS, rows))
    blocks.append(("chart", "scan_latency_vs_length"))

    se = _num(off.get("scan_errors"))
    if se is not None:
        blocks.append(("p", f"Scan errors (excluded from accuracy): {_fmt_int(se)}."))
    if data.get("offline_findings_count") is not None:
        blocks.append(("p", f"Findings evaluated: {_fmt_int(data['offline_findings_count'])} "
                            f"rows in offline_findings.jsonl."))
    return {"id": "offline-accuracy", "title": "Offline accuracy", "blocks": blocks}


def _sec_sweep(data, charts) -> dict:
    blocks: List[tuple] = []
    sweep = (data.get("offline") or {}).get("sweep")
    if not isinstance(sweep, dict):
        blocks.append(("note", _NO_DATA + "no threshold sweep in offline_metrics.json."))
        return {"id": "threshold-sweep", "title": "Threshold sweep", "blocks": blocks}
    best = sweep.get("best_f1") or {}
    pts = [p for p in (sweep.get("points") or []) if isinstance(p, dict)]
    pr_area, roc_area = _sweep_partial_areas(sweep)
    blocks.append(("p", f"Partial PR area {_fmt(pr_area, 3)} over observed recall "
                        f"{_obs_span([p.get('span_recall') for p in pts])} · "
                        f"partial ROC area {_fmt(roc_area, 3)} over observed FPR "
                        f"{_obs_span([p.get('doc_fpr') for p in pts])} · "
                        f"best span-F1 {_fmt(best.get('f1'), 3)} at "
                        f"threshold {_fmt(best.get('threshold'), 2)}."))
    blocks.append(("p", "Areas are trapezoid integrals over the swept thresholds only "
                        "— partial-curve areas, not full-curve AUCs, and not "
                        "comparable to a 0.5 chance level."))
    blocks.append(("chart", "threshold_sweep"))
    if pts:
        rows = [[_fmt(p.get("threshold"), 2), _fmt(p.get("span_precision")),
                 _fmt(p.get("span_recall")), _fmt(p.get("span_f1")),
                 _fmt(p.get("doc_precision")), _fmt(p.get("doc_recall")),
                 _fmt(p.get("doc_specificity")), _fmt(p.get("doc_fpr"))]
                for p in pts[:25]]
        blocks.append(("table",
                       ["threshold", "span P", "span R", "span F1",
                        "doc P", "doc R", "doc spec", "doc FPR"], rows))
        if len(pts) > 25:
            blocks.append(("p", f"(first 25 of {len(pts)} sweep points shown)"))
    return {"id": "threshold-sweep", "title": "Threshold sweep", "blocks": blocks}


def _sec_e2e(data, charts) -> dict:
    blocks: List[tuple] = []
    e2e = data.get("e2e_metrics")
    if not isinstance(e2e, dict):
        blocks.append(("note", _NO_DATA + "no e2e_metrics.json in the supplied run directories."))
        return {"id": "e2e-detection", "title": "E2E detection", "blocks": blocks}

    run = e2e.get("run")
    if isinstance(run, dict) and run:
        blocks.append(("h3", "Run parameters"))
        blocks.append(("table", ["parameter", "value"], _kv_rows(run), "text"))

    send = e2e.get("send")
    if isinstance(send, dict):
        blocks.append(("h3", "Send results"))
        blocks.append(("table", ["sent", "ok", "failed"],
                       [[_fmt_int(send.get("n_sent")), _fmt_int(send.get("n_ok")),
                         _fmt_int(send.get("n_failed"))]]))
        if isinstance(send.get("client_latency_ms"), dict):
            blocks.append(("table", _LAT_HEADERS,
                           [["client latency"] + _lat_cells(send["client_latency_ms"])]))

    cov = e2e.get("coverage")
    if isinstance(cov, dict):
        blocks.append(("h3", "Detection coverage"))
        blocks.append(("table",
                       ["dirty sent", "dirty alerted", "rate",
                        "in-scope sent", "in-scope alerted", "in-scope rate"],
                       [[_fmt_int(cov.get("dirty_sent")), _fmt_int(cov.get("dirty_alerted")),
                         _fmt_pct(cov.get("rate")),
                         _fmt_int(cov.get("in_scope_dirty_sent")),
                         _fmt_int(cov.get("in_scope_dirty_alerted")),
                         _fmt_pct(cov.get("in_scope_rate"))]]))
    cfp = e2e.get("clean_fp")
    if isinstance(cfp, dict):
        blocks.append(("table", ["clean sent", "clean alerted", "FP rate"],
                       [[_fmt_int(cfp.get("clean_sent")), _fmt_int(cfp.get("clean_alerted")),
                         _fmt_pct(cfp.get("rate"))]]))

    pcd = e2e.get("per_category_detection")
    if isinstance(pcd, dict) and pcd:
        blocks.append(("h3", "Per-category detection"))
        blocks.append(("table", ["category", "expected", "detected", "recall"],
                       [[c, _fmt_int((r or {}).get("expected")),
                         _fmt_int((r or {}).get("detected")),
                         _fmt_pct((r or {}).get("recall"))]
                        for c, r in pcd.items()]))

    blocks.extend(_severity_blocks(e2e.get("severity"), "Severity confusion (expected vs alert)"))

    sc = e2e.get("scanner_counts")
    t = _count_table(sc, "scanner")
    if t:
        blocks.append(("h3", "Alerts by scanner"))
        blocks.append(t)

    lat_rows = [[name] + _lat_cells(e2e.get(key))
                for key, name in (("scan_latency_ms", "scan latency"),
                                  ("scan_lag_ms", "scan lag"))
                if isinstance(e2e.get(key), dict)]
    if lat_rows:
        blocks.append(("h3", "Scan timing (ms)"))
        blocks.append(("table", _LAT_HEADERS, lat_rows))

    drain = e2e.get("drain")
    if isinstance(drain, dict):
        blocks.append(("p", f"Drain: {_fmt(drain.get('seconds'), 1)}s, "
                            f"settled={drain.get('settled')}."))
    if _num(e2e.get("scanner_error_alerts")) is not None:
        blocks.append(("p", f"Scanner-error alerts (excluded from accuracy): "
                            f"{_fmt_int(e2e.get('scanner_error_alerts'))}."))
    cleanup = e2e.get("cleanup")
    if isinstance(cleanup, dict) and cleanup.get("alerts_purged") is not None:
        blocks.append(("p", f"Cleanup: {_fmt_int(cleanup.get('alerts_purged'))} "
                            f"harness alert rows purged."))
    if data.get("e2e_results_count") is not None:
        blocks.append(("p", f"Per-document results: {_fmt_int(data['e2e_results_count'])} "
                            f"rows in e2e_results.jsonl."))
    return {"id": "e2e-detection", "title": "E2E detection", "blocks": blocks}


def _sec_load(data, charts) -> dict:
    blocks: List[tuple] = []
    phases = [p for p in (data.get("load_phases") or []) if isinstance(p, dict)]
    if not phases:
        blocks.append(("note", _NO_DATA + "no load_phases.json in the supplied run directories."))
        return {"id": "load-overhead", "title": "Load & overhead", "blocks": blocks}

    rows = []
    for ph in phases:
        off_, dlp = ph.get("offered") or {}, ph.get("dlp") or {}
        e2e_lat = (ph.get("latency_ms") or {}).get("e2e") or {}
        lag = dlp.get("scan_lag_ms") or {}
        cpu = ph.get("cpu") or {}
        rows.append([
            str(ph.get("phase_id")), str(ph.get("scanner_mode")),
            _fmt_int(ph.get("concurrency")), _fmt(ph.get("duration_s"), 0),
            _fmt_int(off_.get("n_requests")), _fmt_int(off_.get("n_err")),
            _fmt(off_.get("rps"), 1), _fmt_ms(e2e_lat.get("p50")),
            _fmt_ms(e2e_lat.get("p95")), _fmt_pct(dlp.get("coverage_rate")),
            _fmt_int(dlp.get("alerts")), _fmt_ms(lag.get("p95")),
            _fmt(dlp.get("drain_seconds"), 1), _fmt_int(dlp.get("queue_drops_logged")),
            _fmt(cpu.get("app_mean_pct"), 1),
        ])
    blocks.append(("h3", "Phases"))
    blocks.append(("table",
                   ["phase", "mode", "conc", "dur (s)", "reqs", "err", "rps",
                    "e2e p50", "e2e p95", "coverage", "alerts", "lag p95",
                    "drain (s)", "drops", "cpu %"], rows))
    for name in ("load_e2e_latency", "load_throughput", "load_coverage",
                 "load_scan_lag", "load_drain", "load_cpu"):
        blocks.append(("chart", name))

    bc = [r for r in (data.get("baseline_comparison") or []) if isinstance(r, dict)]
    if bc:
        blocks.append(("h3", "Baseline comparison (vs scanners off)"))
        blocks.append(("table",
                       ["concurrency", "mode", "e2e p50 Δ (ms)", "e2e p95 Δ (ms)",
                        "throughput Δ (%)", "coverage"],
                       [[_fmt_int(r.get("concurrency")), str(r.get("mode")),
                         _fmt_ms(r.get("e2e_p50_delta_ms")), _fmt_ms(r.get("e2e_p95_delta_ms")),
                         _fmt(r.get("throughput_delta_pct"), 1),
                         _fmt_pct(r.get("coverage_rate"))] for r in bc]))
    if data.get("load_requests_count"):
        blocks.append(("p", f"Per-request rows: {_fmt_int(data['load_requests_count'])} "
                            f"in load_requests.jsonl; CPU samples: "
                            f"{_fmt_int(data.get('cpu_samples_count'))}."))
    return {"id": "load-overhead", "title": "Load & overhead", "blocks": blocks}


def _sec_pipeline(data) -> dict:
    blocks: List[tuple] = [
        ("p", f"DLP scanning is post-hoc: each request id is enqueued to a per-worker "
              f"asyncio queue (maxsize {DLP_QUEUE_MAXSIZE:,}) with a single serial "
              f"consumer; on overflow the id is dropped silently ('dlp_queue_full' "
              f"log line) and the document is never scanned. Clean scans write no row, "
              f"so coverage is only measurable against known-dirty documents."),
    ]
    e2e = data.get("e2e_metrics") or {}
    have_any = False
    if isinstance(e2e.get("scan_lag_ms"), dict) or isinstance(e2e.get("drain"), dict):
        have_any = True
        drain = e2e.get("drain") or {}
        blocks.append(("h3", "E2E pipeline"))
        rows = []
        if isinstance(e2e.get("scan_lag_ms"), dict):
            rows.append(["scan lag (ms)"] + _lat_cells(e2e["scan_lag_ms"]))
        if rows:
            blocks.append(("table", _LAT_HEADERS, rows))
        blocks.append(("p", f"Drain {_fmt(drain.get('seconds'), 1)}s "
                            f"(settled={drain.get('settled')}); scanner-error alerts: "
                            f"{_fmt_int(e2e.get('scanner_error_alerts'))}."))
    phases = [p for p in (data.get("load_phases") or []) if isinstance(p, dict)]
    if phases:
        have_any = True
        blocks.append(("h3", "Load pipeline health by phase"))
        blocks.append(("table",
                       ["phase", "mode", "queue drops", "drain (s)", "settled",
                        "scanner errors"],
                       [[str(p.get("phase_id")), str(p.get("scanner_mode")),
                         _fmt_int((p.get("dlp") or {}).get("queue_drops_logged")),
                         _fmt((p.get("dlp") or {}).get("drain_seconds"), 1),
                         str((p.get("dlp") or {}).get("drain_settled")),
                         _fmt_int((p.get("dlp") or {}).get("scanner_error_alerts"))]
                        for p in phases]))
    if not have_any:
        blocks.append(("note", _NO_DATA + "no e2e or load artifacts to assess pipeline health."))
    return {"id": "pipeline-health", "title": "Scan pipeline health", "blocks": blocks}


def _sec_config(data) -> dict:
    blocks: List[tuple] = []
    runs = data.get("runs") or []
    rows = []
    for r in runs:
        m = r.get("manifest") or {}
        rows.append([os.path.basename(r["dir"]), str(m.get("kind", "—")),
                     str(m.get("created_at", "—")), str(m.get("scanner_mode", "—")),
                     str(m.get("seed", "—")), str(m.get("git_rev", "—"))])
    if rows:
        blocks.append(("h3", "Run directories"))
        blocks.append(("table", ["run", "kind", "created", "scanner mode", "seed", "git rev"],
                       rows, "text"))
    snap = data.get("config_snapshot")
    if isinstance(snap, dict) and snap:
        blocks.append(("h3", "DLP configuration snapshot"))
        blocks.append(("table", ["key", "value"],
                       [[k, v if isinstance(v, str) else json.dumps(v)]
                        for k, v in sorted(snap.items())], "text"))
    else:
        blocks.append(("note", _NO_DATA + "no config_snapshot.json in the supplied run directories."))
    return {"id": "config-appendix", "title": "Configuration appendix", "blocks": blocks}


def _sec_methodology() -> dict:
    paras = [
        "Lenient span matching counts a ground-truth entity as detected when any "
        "finding overlaps its span (or, for span-less scanners, contains its text); "
        "strict matching additionally requires the finding's canonical category to "
        "match, so a detected-but-mislabeled entity counts as a miss.",
        "In-scope recall restricts the denominator to categories the active scanner "
        "configuration is capable of detecting — the fair per-scanner number. System "
        "recall counts every planted entity — the honest number a user of the whole "
        "pipeline experiences.",
        "Synthetic scanner-failure alerts (categories=[\"dlp_scanner_error\"]) are "
        "excluded from all accuracy math and reported separately as degraded scans.",
        "Confidence intervals are 95% percentile bootstrap over whole documents "
        "(entities within a document are correlated, so the document is the "
        "resampling unit).",
        "Coverage = alerted dirty documents / sent dirty documents; clean-FP rate = "
        "alerted clean documents / sent clean documents. Scan lag measures request "
        "completion to scanned_at; drain measures send-stop until the alert table "
        "settles. Latency percentiles are linear-interpolated.",
    ]
    return {"id": "methodology", "title": "Methodology",
            "blocks": [("p", t) for t in paras]}


def _build_sections(data, recs, charts) -> List[dict]:
    return [
        _sec_executive(data, recs, charts),
        _sec_corpus(data),
        _sec_offline(data, charts),
        _sec_sweep(data, charts),
        _sec_e2e(data, charts),
        _sec_load(data, charts),
        _sec_pipeline(data),
        _sec_config(data),
        _sec_methodology(),
    ]


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _md_cell(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _md_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    out = ["| " + " | ".join(_md_cell(h) for h in headers) + " |",
           "|" + "|".join(" --- " for _ in headers) + "|"]
    out += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in rows]
    out.append("")
    return out


def _render_markdown(title: str, sections: List[dict],
                     charts: Dict[str, dict]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# {title}", "", f"_Generated {ts} by dlp_harness.report._", ""]
    for sec in sections:
        lines += [f"## {sec['title']}", ""]
        for block in sec["blocks"]:
            kind = block[0]
            if kind == "h3":
                lines += [f"### {block[1]}", ""]
            elif kind in ("p",):
                lines += [block[1], ""]
            elif kind == "note":
                lines += [f"> {block[1]}", ""]
            elif kind == "pre":
                lines += ["```json", block[1], "```", ""]
            elif kind == "table":
                lines += _md_table(block[1], block[2])
            elif kind == "tiles":
                lines += _md_table(["metric", "value", "context"],
                                   [[l, v, s] for l, v, s in block[1]])
            elif kind == "recs":
                lines += _md_table(["severity", "finding", "recommendation"],
                                   [[r["severity"], r["finding"], r["recommendation"]]
                                    for r in block[1]])
            elif kind == "chart":
                c = charts.get(block[1])
                if c:
                    lines += [f"![{c['caption']}]({c['file']})", "",
                              f"_{c['caption']}_", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML renderer (single self-contained file, light theme, U of I gold accent)
# ---------------------------------------------------------------------------

_CSS = """
:root { --gold: #F1B300; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
        --grid: #e1e0d9; --line: #e6e4dd; --card: #ffffff; --page: #f7f6f3; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 0 20px 60px; }
header { border-top: 6px solid var(--gold); background: var(--card);
         border-bottom: 1px solid var(--line); margin-bottom: 20px;
         padding: 26px 0 18px; }
header .wrap { padding-bottom: 0; }
h1 { margin: 0 0 4px; font-size: 26px; }
.sub { color: var(--ink2); font-size: 13px; }
nav { margin-top: 12px; font-size: 13px; }
nav a { color: var(--ink2); text-decoration: none; margin-right: 14px; }
nav a:hover { color: var(--ink); border-bottom: 2px solid var(--gold); }
section { background: var(--card); border: 1px solid var(--line);
          border-radius: 10px; padding: 20px 24px 24px; margin: 18px 0; }
h2 { font-size: 20px; margin: 0 0 14px; padding-bottom: 8px;
     border-bottom: 2px solid var(--gold); }
h3 { font-size: 15px; margin: 20px 0 8px; color: var(--ink); }
p { margin: 8px 0; color: var(--ink); }
.note { color: var(--ink2); background: #faf8f2; border-left: 3px solid var(--gold);
        padding: 8px 12px; border-radius: 0 6px 6px 0; font-size: 14px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; margin: 8px 0; }
th { text-align: left; color: var(--ink2); font-weight: 600; font-size: 12px;
     border-bottom: 1.5px solid #c3c2b7; padding: 6px 10px; white-space: nowrap; }
td { border-bottom: 1px solid var(--grid); padding: 6px 10px; vertical-align: top; }
table.num td:not(:first-child), table.num th:not(:first-child)
  { text-align: right; font-variant-numeric: tabular-nums; }
pre { background: #faf9f6; border: 1px solid var(--line); border-radius: 8px;
      padding: 12px; font-size: 12px; overflow-x: auto; }
figure { margin: 14px 0; border: 1px solid var(--line); border-radius: 8px;
         padding: 12px; background: var(--card); }
figure img { max-width: 100%; height: auto; display: block; }
figcaption { color: var(--muted); font-size: 12px; margin-top: 8px; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 4px 0 8px; }
.tile { border: 1px solid var(--line); border-left: 3px solid var(--gold);
        border-radius: 8px; padding: 12px 16px; min-width: 168px;
        background: #fcfcfb; }
.tile .v { font-size: 25px; font-weight: 600; }
.tile .l { font-size: 12px; color: var(--ink2); }
.tile .s { font-size: 11px; color: var(--muted); }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
       margin-right: 6px; vertical-align: baseline; }
.sev { white-space: nowrap; font-weight: 600; font-size: 12px; }
footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
"""


def _html_table(headers, rows, cls="num") -> str:
    esc = _htmllib.escape
    head = "".join(f"<th>{esc(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in row) + "</tr>"
                   for row in rows)
    return (f'<div class="scroll"><table class="{cls}">'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def _html_block(block, charts) -> str:
    esc = _htmllib.escape
    kind = block[0]
    if kind == "h3":
        return f"<h3>{esc(block[1])}</h3>"
    if kind == "p":
        return f"<p>{esc(block[1])}</p>"
    if kind == "note":
        return f'<p class="note">{esc(block[1])}</p>'
    if kind == "pre":
        return f"<pre>{esc(block[1])}</pre>"
    if kind == "table":
        cls = block[3] if len(block) > 3 else "num"
        return _html_table(block[1], block[2], "num" if cls == "num" else "text")
    if kind == "tiles":
        tiles = "".join(
            f'<div class="tile"><div class="l">{esc(l)}</div>'
            f'<div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>'
            for l, v, s in block[1])
        return f'<div class="tiles">{tiles}</div>'
    if kind == "recs":
        rows = "".join(
            "<tr><td><span class='sev'>"
            f"<span class='dot' style='background:{_SEV_COLORS.get(r['severity'], _MUTED)}'></span>"
            f"{esc(r['severity'])}</span></td>"
            f"<td>{esc(r['finding'])}</td><td>{esc(r['recommendation'])}</td></tr>"
            for r in block[1])
        return ('<div class="scroll"><table class="text"><thead><tr>'
                "<th>severity</th><th>finding</th><th>recommendation</th>"
                f"</tr></thead><tbody>{rows}</tbody></table></div>")
    if kind == "chart":
        c = charts.get(block[1])
        if not c:
            return ""
        return (f'<figure><img alt="{esc(c["caption"])}" '
                f'src="data:image/png;base64,{c["b64"]}">'
                f"<figcaption>{esc(c['caption'])}</figcaption></figure>")
    return ""


def _render_html(title: str, sections: List[dict], charts: Dict[str, dict]) -> str:
    esc = _htmllib.escape
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nav = "".join(f'<a href="#{s["id"]}">{esc(s["title"])}</a>' for s in sections)
    body = []
    for sec in sections:
        blocks = "".join(_html_block(b, charts) for b in sec["blocks"])
        body.append(f'<section id="{sec["id"]}"><h2>{esc(sec["title"])}</h2>{blocks}</section>')
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f'<header><div class="wrap"><h1>{esc(title)}</h1>'
        f'<div class="sub">Generated {ts} by dlp_harness.report</div>'
        f"<nav>{nav}</nav></div></header>\n"
        f'<div class="wrap">{"".join(body)}'
        f"<footer>MindRouter DLP evaluation harness — report generated {ts}.</footer>"
        "</div>\n</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_report(run_dirs: List[str], out_dir: str,
                    title: str = "MindRouter DLP Evaluation") -> Dict[str, Any]:
    """Render run-directory artifacts into report.md + report.html + charts.

    Returns {"md_path", "html_path", "charts": [png paths]}. Never raises on
    missing or malformed artifacts — those degrade to notes and warnings.
    """
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
    data = load_run_data(list(run_dirs))
    os.makedirs(out_dir, exist_ok=True)
    charts = _build_charts(data, out_dir)
    recs = build_recommendations(data)
    sections = _build_sections(data, recs, charts)

    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(title, sections, charts))
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_html(title, sections, charts))
    return {"md_path": md_path, "html_path": html_path,
            "charts": [c["path"] for c in charts.values()]}


# ---------------------------------------------------------------------------
# PDF rendering (optional, macOS/Linux with a Chromium-family browser)
# ---------------------------------------------------------------------------

_CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def find_chrome() -> Optional[str]:
    """Path to a headless-capable Chromium-family browser, or None."""
    for p in _CHROME_CANDIDATES:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return None


def html_to_pdf(html_path: str, pdf_path: str, chrome: Optional[str] = None,
                timeout_s: float = 120.0) -> str:
    """Print the self-contained report HTML to PDF via headless Chrome.

    The HTML embeds every chart as a data: URI, so file:// printing needs no
    network. Raises RuntimeError when no browser is available or the print
    fails — callers treat PDF as an optional extra, never a report blocker.
    """
    import subprocess
    chrome = chrome or find_chrome()
    if not chrome:
        raise RuntimeError(
            "no Chromium-family browser found for PDF rendering "
            f"(searched: {', '.join(_CHROME_CANDIDATES)})")
    result = subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={os.path.abspath(pdf_path)}",
         f"file://{os.path.abspath(html_path)}"],
        capture_output=True, timeout=timeout_s)
    if result.returncode != 0 or not os.path.exists(pdf_path):
        tail = (result.stderr or b"")[-300:].decode("utf-8", "replace")
        raise RuntimeError(f"headless chrome PDF print failed: {tail}")
    return os.path.abspath(pdf_path)
