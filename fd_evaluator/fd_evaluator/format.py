"""Pretty-print compute_metrics() results in grouped sections."""

from __future__ import annotations


def format_results(results: dict[str, float]) -> str:
    """Group flat metric dict into FID, FDr, MIND sections."""
    sections: list[tuple[str, list[tuple[str, float]]]] = []

    fid_rows = []
    for k in ("fid", "inception_score"):
        if k in results:
            fid_rows.append((k, results[k]))
    if fid_rows:
        sections.append(("FID / Inception Score", fid_rows))

    fdr_keys = sorted(k for k in results if k.startswith("fdr_"))
    fdr_rows = [(k, results[k]) for k in fdr_keys]
    if "fdr6" in results:
        fdr_rows.append(("fdr6 (mean)", results["fdr6"]))
    if fdr_rows:
        sections.append(("FDr (lower better)", fdr_rows))

    mind_keys = sorted(k for k in results if k.startswith("mind_"))
    mind_rows = [(k, results[k]) for k in mind_keys]
    if mind_rows:
        sections.append(("MIND", mind_rows))

    lines: list[str] = []
    width = max((len(k) for sec in sections for k, _ in sec[1]), default=12)
    for title, rows in sections:
        bar = "-" * (width + 14)
        lines.append(bar)
        lines.append(f"  {title}")
        lines.append(bar)
        for k, v in rows:
            lines.append(f"  {k:<{width}}  {v:>10.4f}")
        lines.append("")
    return "\n".join(lines).rstrip()
