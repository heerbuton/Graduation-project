#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Derive ablation metrics using A2 as pseudo ground truth.
Outputs:
  - results/a2_reference_summary.json
  - results/a2_reference_summary.csv
  - results/a2_reference_report.md
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_note_map(notes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for note in notes:
        if not isinstance(note, dict):
            continue
        gid = str(note.get("group_id", "")).strip()
        if not gid:
            continue
        mapping[gid] = note
    return mapping


def main() -> int:
    exp_root = Path(__file__).resolve().parents[1]
    results_dir = exp_root / "results"
    variants = ["A0", "A1", "A2"]

    page_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_pages = set()
    for variant in variants:
        page_data[variant] = {}
        pages_dir = results_dir / variant / "pages"
        for p in sorted(pages_dir.glob("page_*.result.json")):
            data = read_json(p)
            pid = str(data.get("page_id", p.stem))
            page_data[variant][pid] = data
            all_pages.add(pid)

    all_pages = sorted(all_pages)

    # A2 success pages define pseudo-ground-truth scope.
    a2_success_pages = [
        pid for pid in all_pages if page_data.get("A2", {}).get(pid, {}).get("inference", {}).get("ok") is True
    ]
    a2_failed_pages = [
        pid for pid in all_pages if pid not in a2_success_pages
    ]

    # Count A2 note totals.
    a2_total_groups_success_scope = 0
    for pid in a2_success_pages:
        a2_notes = page_data["A2"][pid].get("notes_merged", [])
        a2_total_groups_success_scope += len(a2_notes)

    output: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "A2 as pseudo ground truth",
        "scope": {
            "all_pages_count": len(all_pages),
            "a2_success_pages_count": len(a2_success_pages),
            "a2_failed_pages_count": len(a2_failed_pages),
            "a2_failed_pages": a2_failed_pages,
            "a2_total_groups_success_scope": a2_total_groups_success_scope,
        },
        "variants": {},
    }

    field_names = ["pitch", "octave", "duration", "new_measure"]

    for variant in variants:
        v_page_present_success = 0
        v_page_full_match = 0
        v_group_coverage = 0
        v_group_full_match = 0
        v_field_match = {name: 0 for name in field_names}
        v_field_total = 0
        missing_pages = []
        page_rows = []

        for pid in a2_success_pages:
            a2_rec = page_data["A2"][pid]
            a2_notes = a2_rec.get("notes_merged", [])
            a2_map = as_note_map(a2_notes)
            a2_group_ids = list(a2_map.keys())

            v_rec = page_data.get(variant, {}).get(pid)
            if not v_rec or v_rec.get("inference", {}).get("ok") is not True:
                missing_pages.append(
                    {
                        "page_id": pid,
                        "reason": (v_rec or {}).get("inference", {}).get("error", "missing or infer_not_ok"),
                        "a2_group_count": len(a2_group_ids),
                    }
                )
                page_rows.append(
                    {
                        "page_id": pid,
                        "a2_group_count": len(a2_group_ids),
                        "variant_infer_ok": False,
                        "covered_groups": 0,
                        "full_match_groups": 0,
                        "page_full_match": False,
                    }
                )
                continue

            v_page_present_success += 1
            v_notes = v_rec.get("notes_merged", [])
            v_map = as_note_map(v_notes)

            covered_groups = 0
            full_match_groups = 0
            page_full_match = True

            for gid in a2_group_ids:
                a2_note = a2_map[gid]
                v_note = v_map.get(gid)
                if v_note is None:
                    page_full_match = False
                    continue

                covered_groups += 1
                note_full_match = True
                for fname in field_names:
                    a_val = a2_note.get(fname, False if fname == "new_measure" else "")
                    v_val = v_note.get(fname, False if fname == "new_measure" else "")
                    v_field_total += 1
                    if a_val == v_val:
                        v_field_match[fname] += 1
                    else:
                        note_full_match = False

                if note_full_match:
                    full_match_groups += 1
                else:
                    page_full_match = False

            if covered_groups != len(a2_group_ids):
                page_full_match = False

            if page_full_match:
                v_page_full_match += 1

            v_group_coverage += covered_groups
            v_group_full_match += full_match_groups

            page_rows.append(
                {
                    "page_id": pid,
                    "a2_group_count": len(a2_group_ids),
                    "variant_infer_ok": True,
                    "covered_groups": covered_groups,
                    "full_match_groups": full_match_groups,
                    "page_full_match": page_full_match,
                }
            )

        page_success_rate = (
            v_page_present_success / len(a2_success_pages) if a2_success_pages else 0.0
        )
        page_full_match_rate = (
            v_page_full_match / len(a2_success_pages) if a2_success_pages else 0.0
        )
        group_coverage_rate = (
            v_group_coverage / a2_total_groups_success_scope if a2_total_groups_success_scope else 0.0
        )
        group_full_match_rate = (
            v_group_full_match / a2_total_groups_success_scope if a2_total_groups_success_scope else 0.0
        )

        output["variants"][variant] = {
            "page_success_vs_a2_scope": {
                "success_pages": v_page_present_success,
                "a2_success_pages": len(a2_success_pages),
                "rate": round(page_success_rate, 6),
            },
            "page_full_match_vs_a2": {
                "full_match_pages": v_page_full_match,
                "a2_success_pages": len(a2_success_pages),
                "rate": round(page_full_match_rate, 6),
            },
            "group_coverage_vs_a2": {
                "covered_groups": v_group_coverage,
                "a2_groups": a2_total_groups_success_scope,
                "rate": round(group_coverage_rate, 6),
            },
            "group_full_match_vs_a2": {
                "full_match_groups": v_group_full_match,
                "a2_groups": a2_total_groups_success_scope,
                "rate": round(group_full_match_rate, 6),
            },
            "field_match_vs_a2": {
                "total_compared": v_field_total,
                **{
                    f"{fname}_match": v_field_match[fname]
                    for fname in field_names
                },
                **{
                    f"{fname}_rate": (
                        round(v_field_match[fname] / (v_group_coverage or 1), 6)
                        if v_group_coverage
                        else 0.0
                    )
                    for fname in field_names
                },
            },
            "missing_or_failed_pages_in_a2_scope": missing_pages,
            "per_page_rows": page_rows,
        }

    json_out = results_dir / "a2_reference_summary.json"
    json_out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_lines = [
        "variant,page_success_rate,page_full_match_rate,group_coverage_rate,group_full_match_rate,pitch_rate,octave_rate,duration_rate,new_measure_rate"
    ]
    for variant in variants:
        v = output["variants"][variant]
        csv_lines.append(
            ",".join(
                [
                    variant,
                    str(v["page_success_vs_a2_scope"]["rate"]),
                    str(v["page_full_match_vs_a2"]["rate"]),
                    str(v["group_coverage_vs_a2"]["rate"]),
                    str(v["group_full_match_vs_a2"]["rate"]),
                    str(v["field_match_vs_a2"]["pitch_rate"]),
                    str(v["field_match_vs_a2"]["octave_rate"]),
                    str(v["field_match_vs_a2"]["duration_rate"]),
                    str(v["field_match_vs_a2"]["new_measure_rate"]),
                ]
            )
        )
    (results_dir / "a2_reference_summary.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    md_lines = [
        "# A2 参考真值派生结果（Pseudo Ground Truth）",
        "",
        f"- generated_at: {output['generated_at']}",
        f"- a2_success_pages: {len(a2_success_pages)}",
        f"- a2_groups_in_scope: {a2_total_groups_success_scope}",
        "",
        "## 三方案相对 A2 的一致性",
        "",
    ]
    for variant in variants:
        v = output["variants"][variant]
        md_lines.append(
            (
                f"- {variant}: "
                f"page_success={v['page_success_vs_a2_scope']['success_pages']}/{v['page_success_vs_a2_scope']['a2_success_pages']} "
                f"({v['page_success_vs_a2_scope']['rate']}); "
                f"group_full_match={v['group_full_match_vs_a2']['full_match_groups']}/{v['group_full_match_vs_a2']['a2_groups']} "
                f"({v['group_full_match_vs_a2']['rate']}); "
                f"pitch/oct/dur/new={v['field_match_vs_a2']['pitch_rate']}/"
                f"{v['field_match_vs_a2']['octave_rate']}/"
                f"{v['field_match_vs_a2']['duration_rate']}/"
                f"{v['field_match_vs_a2']['new_measure_rate']}"
            )
        )

    md_lines.extend(
        [
            "",
            "## 备注",
            "",
            "- 本文件采用 A2 结果作为参考真值，因此 A2 对自身一致率天然为 100%。",
            "- 该结果可用于“相对一致性”展示，不等同于外部人工标注真值准确率。",
        ]
    )
    (results_dir / "a2_reference_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(str(json_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

