"""Plain-text report for the CLI."""
from __future__ import annotations

DISCLAIMER = "Model output for research only, not investment advice."


def to_text(rep: dict) -> str:
    L = [f"== {rep['name']} =="]
    for w in rep["warnings"]:
        L.append(f"! {w}")

    L.append("\nSHORT TERM (listing and the first weeks)")
    if rep["missing_short_term"]:
        L.append(f"  Cannot estimate yet. Missing: {', '.join(rep['missing_short_term'])}")
        L.append("  Provide them with --set key=value (e.g. --set gmp_final=221).")
    else:
        L.append(f"  Raw GMP implies:        {rep['gmp_pct']:+.1f}%")
        L.append(f"  Heuristic estimate:     {rep['heuristic_gain']:+.1f}%")
        if rep.get("ml"):
            m = rep["ml"]
            L.append(f"  ML estimate ({m['tier']}):    {m['q50']:+.1f}%  (80% band {m['q10']:+.1f}% to {m['q90']:+.1f}%)")
        L.append(f"  Headline ({rep['basis']}): {rep['headline_gain']:+.1f}%  ->  {rep['short_verdict']}")
        for s in rep.get("scenarios", []):
            L.append(f"    {s['scenario']:<55} {s['est_gain_pct']:+5.1f}%  {s['call']}")
        for n in rep["notes"]:
            L.append(f"  Note: {n}")

    L.append("\nLONG TERM (1-3 years)")
    lt = rep.get("long_term")
    if lt:
        score = f"{lt['score']:.0f}/100" if lt["score"] is not None else "N/A"
        L.append(f"  Score {score}  ->  {lt['verdict']}")
        L.append(f"  Fundamental coverage: {lt['coverage_pct']}%")
        for k, val in lt["parts"].items():
            L.append(f"    {k:<28} {val}/10")
        for item in lt["evidence"]:
            L.append(f"  Evidence: {item}")
        for item in lt["unusable_fields"]:
            L.append(f"  Not usable: {item}")
        if lt["missing_fields"]:
            L.append(f"  Still needed: {'; '.join(lt['missing_field_descriptions'])}")
    else:
        L.append(f"  Not scored. Missing: {', '.join(rep.get('long_term_missing', []))}")

    L.append("\nDATA USED (field, value, source, fetched at)")
    for r in rep["provenance"]:
        L.append(f"  {r['field']:<36} {str(r['value']):<12} {r['source']}  [{r['fetched_at']}]")
    L.append(f"\n{DISCLAIMER}")
    return "\n".join(L)
