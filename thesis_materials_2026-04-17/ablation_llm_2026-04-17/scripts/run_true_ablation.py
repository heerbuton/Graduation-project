#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
True ablation runner for LLM notation module.
Runs three variants on a fixed topology dataset without touching production code:
- A0: no tone-table injection, no repair
- A1: tone-table injection, no repair
- A2: full pipeline (tone-table + strict validation + repair)

All artifacts are written inside the experiment folder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VARIANTS = [
    {
        "id": "A0",
        "name": "base_no_tone_no_repair",
        "tone_injection": False,
        "repair": False,
        "description": "无音位表、无修复，仅主生成 + 严格合并校验",
    },
    {
        "id": "A1",
        "name": "tone_no_repair",
        "tone_injection": True,
        "repair": False,
        "description": "有音位表、无修复",
    },
    {
        "id": "A2",
        "name": "full_tone_validate_repair",
        "tone_injection": True,
        "repair": True,
        "description": "完整方案：音位表 + 本地校验 + 定向修复",
    },
]


def now_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    last_err: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"读取 JSON 失败: {path} ({last_err})")


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    while True:
        if (current / "backend").exists() and (current / "test").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    raise RuntimeError("无法自动定位仓库根目录（未找到 backend/ 与 test/）。")


def load_module(module_path: Path, module_alias: str):
    spec = importlib.util.spec_from_file_location(module_alias, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def looks_like_topology_dict(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    keys = [str(k) for k in obj.keys()]
    return any(k.startswith("group_") for k in keys)


def extract_topology_payload(raw: Any) -> Optional[Any]:
    if isinstance(raw, dict):
        if "topology_json" in raw:
            return raw.get("topology_json")
        data = raw.get("data")
        if isinstance(data, dict) and "topology_json" in data:
            return data.get("topology_json")
        if looks_like_topology_dict(raw):
            return raw
    if isinstance(raw, list):
        return raw
    return None


def discover_candidate_files(repo_root: Path) -> List[Path]:
    candidates: List[Path] = []

    uploads_dir = repo_root / "backend" / "static" / "uploads"
    if uploads_dir.exists():
        candidates.extend(sorted(uploads_dir.glob("*_result.json")))
        candidates.extend(sorted(uploads_dir.glob("*_result_step1.json")))

    test_dir = repo_root / "test"
    if test_dir.exists():
        candidates.extend(sorted(test_dir.rglob("*topology*.json")))

    paper_input = (
        repo_root
        / "paper_assets"
        / "llm_notation_2026-04-03"
        / "01_input_data_spec"
    )
    if paper_input.exists():
        candidates.extend(sorted(paper_input.rglob("*topology*.json")))

    thesis_json = repo_root / "thesis_materials_2026-04-03" / "05_论文可用JSON与XML"
    if thesis_json.exists():
        candidates.extend(sorted(thesis_json.rglob("*topology*.json")))

    deduped: Dict[str, Path] = {}
    for p in candidates:
        deduped[str(p.resolve())] = p.resolve()

    return sorted(deduped.values(), key=lambda x: str(x).lower())


def prepare_fixed_dataset(exp_root: Path, repo_root: Path, llm_module_path: Path) -> Dict[str, Any]:
    data_dir = exp_root / "data"
    fixed_topology_dir = data_dir / "fixed_topologies"
    fixed_topology_dir.mkdir(parents=True, exist_ok=True)

    llm_mod = load_module(llm_module_path, "llm_mod_manifest")

    candidates = discover_candidate_files(repo_root)
    manifest_entries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    seen_hash: Dict[str, str] = {}

    page_idx = 0
    for file_path in candidates:
        try:
            raw = read_json(file_path)
            topology = extract_topology_payload(raw)
            if topology is None:
                skipped.append(
                    {
                        "path": str(file_path),
                        "reason": "未识别到 topology_json 结构",
                    }
                )
                continue

            groups = llm_mod._normalize_topology(topology)  # noqa: SLF001
            if not groups:
                skipped.append(
                    {
                        "path": str(file_path),
                        "reason": "归一化后 group 为空",
                    }
                )
                continue

            canonical = json.dumps(groups, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
            if digest in seen_hash:
                skipped.append(
                    {
                        "path": str(file_path),
                        "reason": f"与 {seen_hash[digest]} 拓扑重复（hash 去重）",
                    }
                )
                continue

            page_idx += 1
            page_id = f"page_{page_idx:03d}"
            out_topology = fixed_topology_dir / f"{page_id}.topology.json"
            write_json(out_topology, topology)

            entry = {
                "page_id": page_id,
                "source_path": str(file_path),
                "copied_topology_path": str(out_topology),
                "group_count": len(groups),
                "topology_hash": digest,
                "first_group_id": groups[0].get("group_id", "") if groups else "",
            }
            manifest_entries.append(entry)
            seen_hash[digest] = str(file_path)
        except Exception as exc:  # noqa: BLE001
            skipped.append(
                {
                    "path": str(file_path),
                    "reason": f"异常: {type(exc).__name__}: {exc}",
                }
            )

    manifest = {
        "created_at": now_str(),
        "repo_root": str(repo_root),
        "llm_baseline": str(llm_module_path),
        "candidate_count": len(candidates),
        "selected_page_count": len(manifest_entries),
        "selected_group_total": int(sum(item["group_count"] for item in manifest_entries)),
        "entries": manifest_entries,
        "skipped": skipped,
    }

    write_json(data_dir / "fixed_dataset_manifest.json", manifest)
    return manifest


def apply_variant_patches(llm_mod, variant: Dict[str, Any]) -> None:
    if not variant["tone_injection"]:

        def compact_groups_no_tone(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            compact: List[Dict[str, Any]] = []
            for item in groups:
                compact_item = {
                    "group_id": str(item.get("group_id", "")).strip(),
                    "right_fingering": str(item.get("right_fingering", "")).strip(),
                    "left_fingering": str(item.get("left_fingering", "")).strip(),
                    "left_finger": str(item.get("left_finger", "")).strip(),
                    "hui": str(item.get("hui", "")).strip(),
                    "xian": str(item.get("xian", "")).strip(),
                    "action": str(item.get("action", "")).strip(),
                    "finger": str(item.get("finger", "")).strip(),
                    "position": str(item.get("position", "")).strip(),
                    "string": str(item.get("string", "")).strip(),
                }
                compact.append(compact_item)
            return compact

        llm_mod._compact_groups_for_prompt = compact_groups_no_tone  # type: ignore[attr-defined] # noqa: SLF001

    if not variant["repair"]:

        def infer_no_repair(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            llm_text = llm_mod._call_llm_with_messages(llm_mod._build_messages(groups))  # noqa: SLF001
            parsed_payload = llm_mod._json_from_text(llm_text)  # noqa: SLF001
            if parsed_payload is None:
                raise ValueError("LLM 返回不是有效 JSON。")

            parsed_notes = llm_mod._extract_note_list(parsed_payload)  # noqa: SLF001
            if not parsed_notes:
                raise ValueError("LLM 返回中没有可解析的 notes。")
            return llm_mod._merge_notes_strict(parsed_notes, groups)  # noqa: SLF001

        llm_mod._infer_full_context_notes = infer_no_repair  # type: ignore[attr-defined] # noqa: SLF001


def validate_structured_notes(llm_mod, groups: List[Dict[str, Any]], notes: Any) -> Dict[str, Any]:
    result = {
        "total_groups": len(groups),
        "notes_count": len(notes) if isinstance(notes, list) else 0,
        "valid_note_count": 0,
        "page_pass": False,
        "errors": [],
    }

    if not isinstance(notes, list):
        result["errors"].append("notes 不是 list")
        return result

    if len(notes) != len(groups):
        result["errors"].append(f"notes 数量与 groups 不一致: notes={len(notes)}, groups={len(groups)}")

    valid_count = 0
    expected_group_ids = [str(item.get("group_id", "")).strip() for item in groups]

    for idx, group_id in enumerate(expected_group_ids):
        if idx >= len(notes):
            break
        note = notes[idx]
        if not isinstance(note, dict):
            continue

        ok = True
        note_group_id = str(note.get("group_id", "")).strip()
        if note_group_id != group_id:
            ok = False

        if str(note.get("pitch", "")).strip() not in llm_mod.VALID_PITCHES:
            ok = False
        if str(note.get("octave", "")).strip() not in llm_mod.VALID_OCTAVES:
            ok = False
        if str(note.get("duration", "")).strip() not in llm_mod.VALID_DURATIONS:
            ok = False

        try:
            llm_mod._normalize_bool_strict(note.get("new_measure", False))  # noqa: SLF001
        except Exception:  # noqa: BLE001
            ok = False

        if ok:
            valid_count += 1

    result["valid_note_count"] = valid_count
    result["page_pass"] = (
        len(groups) > 0 and len(notes) == len(groups) and valid_count == len(groups)
    )
    return result


def convert_to_public_notes(llm_mod, merged_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    public_notes: List[Dict[str, Any]] = []
    for note in merged_notes:
        if not isinstance(note, dict):
            continue
        public_notes.append(llm_mod._public_note(note))  # noqa: SLF001
    return public_notes


def run_variant(
    exp_root: Path,
    variant: Dict[str, Any],
    manifest: Dict[str, Any],
    llm_module_path: Path,
    xml_module_path: Path,
    sleep_between_pages: float,
) -> Dict[str, Any]:
    variant_dir = exp_root / "results" / variant["id"]
    pages_dir = variant_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    llm_mod = load_module(llm_module_path, f"llm_mod_{variant['id']}_{int(time.time()*1000)}")
    xml_mod = load_module(xml_module_path, f"xml_mod_{variant['id']}_{int(time.time()*1000)}")

    apply_variant_patches(llm_mod, variant)

    # Count all LLM requests (including repair calls)
    call_counter = {"count": 0}
    original_call = llm_mod._call_llm_with_messages  # noqa: SLF001

    def wrapped_call(messages: List[Dict[str, str]]) -> str:
        call_counter["count"] += 1
        return original_call(messages)

    llm_mod._call_llm_with_messages = wrapped_call  # type: ignore[attr-defined] # noqa: SLF001

    page_records: List[Dict[str, Any]] = []

    groups_total = 0
    valid_note_total = 0
    structured_pass_pages = 0
    musicxml_success_pages = 0
    infer_failed_pages = 0

    variant_t0 = time.perf_counter()

    for idx, entry in enumerate(manifest["entries"], start=1):
        page_id = entry["page_id"]
        topology_path = Path(entry["copied_topology_path"])
        topology_json = read_json(topology_path)
        groups = llm_mod._normalize_topology(topology_json)  # noqa: SLF001
        groups_total += len(groups)

        infer_start = time.perf_counter()
        notes_merged: List[Dict[str, Any]] = []
        infer_error: Optional[str] = None
        infer_traceback: Optional[str] = None

        try:
            notes_merged = llm_mod._infer_full_context_notes(groups)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            infer_error = f"{type(exc).__name__}: {exc}"
            infer_traceback = traceback.format_exc()
            infer_failed_pages += 1

        infer_elapsed = time.perf_counter() - infer_start

        struct = validate_structured_notes(llm_mod, groups, notes_merged)
        valid_note_total += int(struct["valid_note_count"])
        if struct["page_pass"]:
            structured_pass_pages += 1

        public_notes = convert_to_public_notes(llm_mod, notes_merged)

        musicxml_ok = False
        musicxml_error: Optional[str] = None
        musicxml_traceback: Optional[str] = None
        musicxml_elapsed = 0.0
        musicxml_text = ""

        xml_start = time.perf_counter()
        try:
            musicxml_text = xml_mod.generate_musicxml(public_notes)
            if isinstance(musicxml_text, str) and musicxml_text.strip():
                musicxml_ok = True
                musicxml_success_pages += 1
            else:
                musicxml_error = "generate_musicxml 返回空字符串"
        except Exception as exc:  # noqa: BLE001
            musicxml_error = f"{type(exc).__name__}: {exc}"
            musicxml_traceback = traceback.format_exc()
        finally:
            musicxml_elapsed = time.perf_counter() - xml_start

        page_payload = {
            "page_id": page_id,
            "source_path": entry["source_path"],
            "topology_path": str(topology_path),
            "group_count": len(groups),
            "variant": variant,
            "timing": {
                "infer_s": round(infer_elapsed, 3),
                "musicxml_s": round(musicxml_elapsed, 3),
                "total_s": round(infer_elapsed + musicxml_elapsed, 3),
            },
            "llm_call_count_accumulated": call_counter["count"],
            "structured": struct,
            "inference": {
                "ok": infer_error is None,
                "error": infer_error,
                "traceback": infer_traceback,
            },
            "musicxml": {
                "ok": musicxml_ok,
                "error": musicxml_error,
                "traceback": musicxml_traceback,
                "xml_length": len(musicxml_text or ""),
            },
            "notes_merged": notes_merged,
            "notes_public": public_notes,
        }

        write_json(pages_dir / f"{page_id}.result.json", page_payload)
        if musicxml_text:
            (pages_dir / f"{page_id}.musicxml").write_text(musicxml_text, encoding="utf-8")

        page_records.append(
            {
                "page_id": page_id,
                "group_count": len(groups),
                "infer_ok": infer_error is None,
                "structured_page_pass": bool(struct["page_pass"]),
                "valid_note_count": int(struct["valid_note_count"]),
                "musicxml_ok": musicxml_ok,
                "infer_s": round(infer_elapsed, 3),
                "musicxml_s": round(musicxml_elapsed, 3),
                "total_s": round(infer_elapsed + musicxml_elapsed, 3),
                "inference_error": infer_error,
                "musicxml_error": musicxml_error,
            }
        )

        print(
            f"[{variant['id']}] {idx}/{len(manifest['entries'])} {page_id} | "
            f"groups={len(groups)} struct={struct['page_pass']} musicxml={musicxml_ok} "
            f"infer={infer_elapsed:.2f}s"
        )

        if sleep_between_pages > 0:
            time.sleep(sleep_between_pages)

    variant_elapsed = time.perf_counter() - variant_t0

    pages_total = len(manifest["entries"])
    summary = {
        "variant": variant,
        "created_at": now_str(),
        "page_count": pages_total,
        "group_total": groups_total,
        "valid_note_total": valid_note_total,
        "structured_page_pass_count": structured_pass_pages,
        "musicxml_success_pages": musicxml_success_pages,
        "infer_failed_pages": infer_failed_pages,
        "llm_total_call_count": call_counter["count"],
        "rates": {
            "structured_note_rate": round(valid_note_total / groups_total, 6) if groups_total else 0.0,
            "structured_page_rate": round(structured_pass_pages / pages_total, 6) if pages_total else 0.0,
            "musicxml_page_rate": round(musicxml_success_pages / pages_total, 6) if pages_total else 0.0,
        },
        "runtime": {
            "variant_total_s": round(variant_elapsed, 3),
            "avg_page_s": round(variant_elapsed / pages_total, 3) if pages_total else 0.0,
        },
        "page_records": page_records,
    }

    write_json(variant_dir / "variant_summary.json", summary)
    return summary


def write_final_reports(exp_root: Path, manifest: Dict[str, Any], summaries: List[Dict[str, Any]]) -> None:
    final_json = {
        "created_at": now_str(),
        "manifest": {
            "selected_page_count": manifest["selected_page_count"],
            "selected_group_total": manifest["selected_group_total"],
            "manifest_path": str(exp_root / "data" / "fixed_dataset_manifest.json"),
        },
        "variants": summaries,
    }
    write_json(exp_root / "results" / "ablation_summary.json", final_json)

    # CSV for quick table fill-in
    csv_lines = [
        "variant_id,variant_name,tone_injection,repair,page_count,group_total,valid_note_total,structured_note_rate,structured_page_pass_count,structured_page_rate,musicxml_success_pages,musicxml_page_rate,llm_total_call_count,variant_total_s"
    ]
    for s in summaries:
        v = s["variant"]
        r = s["rates"]
        rt = s["runtime"]
        csv_lines.append(
            ",".join(
                [
                    str(v["id"]),
                    str(v["name"]),
                    str(v["tone_injection"]),
                    str(v["repair"]),
                    str(s["page_count"]),
                    str(s["group_total"]),
                    str(s["valid_note_total"]),
                    str(r["structured_note_rate"]),
                    str(s["structured_page_pass_count"]),
                    str(r["structured_page_rate"]),
                    str(s["musicxml_success_pages"]),
                    str(r["musicxml_page_rate"]),
                    str(s["llm_total_call_count"]),
                    str(rt["variant_total_s"]),
                ]
            )
        )

    (exp_root / "results" / "ablation_summary.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    # Raw markdown note (not polished thesis text)
    md_lines = [
        "# Ablation Raw Results",
        "",
        f"- created_at: {now_str()}",
        f"- selected_pages: {manifest['selected_page_count']}",
        f"- selected_groups: {manifest['selected_group_total']}",
        "",
        "## Variant Lines",
        "",
    ]
    for s in summaries:
        v = s["variant"]
        md_lines.append(
            (
                f"- {v['id']} ({v['name']}): "
                f"结构化合法 {s['valid_note_total']} / {s['group_total']}，"
                f"MusicXML 成功 {s['musicxml_success_pages']} / {s['page_count']}，"
                f"LLM调用总数 {s['llm_total_call_count']}"
            )
        )

    md_lines.extend(
        [
            "",
            "## Files",
            "",
            "- data/fixed_dataset_manifest.json",
            "- results/ablation_summary.json",
            "- results/ablation_summary.csv",
            "- results/A0/pages/*.result.json",
            "- results/A1/pages/*.result.json",
            "- results/A2/pages/*.result.json",
        ]
    )

    (exp_root / "results" / "ablation_raw_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run true ablation for LLM notation module.")
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Experiment root directory",
    )
    parser.add_argument(
        "--sleep-between-pages",
        type=float,
        default=0.0,
        help="Sleep seconds between page requests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_root = args.exp_root.resolve()
    exp_root.mkdir(parents=True, exist_ok=True)

    repo_root = find_repo_root(exp_root)
    snapshot_dir = exp_root / "baseline_snapshot"
    llm_module_path = snapshot_dir / "llm_module.py"
    xml_module_path = snapshot_dir / "musicxml_encoder.py"

    if not llm_module_path.exists() or not xml_module_path.exists():
        raise RuntimeError(
            f"缺少基准快照文件，请确认存在: {llm_module_path} 和 {xml_module_path}"
        )

    run_meta = {
        "created_at": now_str(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "repo_root": str(repo_root),
        "exp_root": str(exp_root),
        "llm_module": str(llm_module_path),
        "musicxml_module": str(xml_module_path),
        "variants": VARIANTS,
    }
    write_json(exp_root / "logs" / "run_meta.json", run_meta)

    print("[1/3] Preparing fixed dataset...")
    manifest = prepare_fixed_dataset(exp_root, repo_root, llm_module_path)
    write_json(exp_root / "logs" / "manifest_snapshot.json", manifest)
    print(
        f"Selected pages={manifest['selected_page_count']} groups={manifest['selected_group_total']} "
        f"from candidates={manifest['candidate_count']}"
    )

    print("[2/3] Running variants...")
    summaries: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        print(f"-- Running {variant['id']} {variant['name']}")
        summary = run_variant(
            exp_root=exp_root,
            variant=variant,
            manifest=manifest,
            llm_module_path=llm_module_path,
            xml_module_path=xml_module_path,
            sleep_between_pages=args.sleep_between_pages,
        )
        summaries.append(summary)
        print(
            f"   done {variant['id']}: structured {summary['valid_note_total']}/{summary['group_total']}, "
            f"musicxml {summary['musicxml_success_pages']}/{summary['page_count']}"
        )

    print("[3/3] Writing final reports...")
    write_final_reports(exp_root, manifest, summaries)

    print("=== FINAL LINES ===")
    for s in summaries:
        v = s["variant"]
        print(
            f"{v['id']}: 结构化合法 {s['valid_note_total']} / {s['group_total']}，"
            f"MusicXML 成功 {s['musicxml_success_pages']} / {s['page_count']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
