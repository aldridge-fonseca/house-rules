"""Judge-report artifacts from a results/<run>/ directory. No network, no paid calls."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOCK_RUN = ROOT / "results" / "MOCK-integration-e2e"


def _load():
    path = ROOT / "scripts" / "make_report.py"
    spec = importlib.util.spec_from_file_location("make_report", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


mr = _load()


def _receipt(system, rid, gold, pred, *, source="openai_moderation", pert="none",
             latency=10, cost=0.001, prefix="omod"):
    return {
        "id": f"{prefix}-{rid}",
        "system": system,
        "text": "t",
        "gold_label": gold,
        "gold_categories": [],
        "source": source,
        "perturbation": pert,
        "prediction": {
            "label": pred, "categories": [], "route": "specialist",
            "latency_ms": latency, "cost_usd": cost,
        },
    }


def _write_run(tmp: Path, systems: dict[str, list[dict]], *, banner: str, modes: dict[str, str]) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    for name, rows in systems.items():
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        (tmp / f"{name}.jsonl").write_text(payload, encoding="utf-8")
    (tmp / "RUN_INFO.json").write_text(
        json.dumps({"banner": banner, "modes": modes}, indent=1),
        encoding="utf-8",
    )
    return tmp


def _fixture(tmp: Path) -> Path:
    specialist = [
        _receipt("specialist-only", "1", "safe", "safe", latency=10, cost=0.001),
        _receipt("specialist-only", "2", "unsafe", "unsafe", latency=30, cost=0.003),
        _receipt("specialist-only", "L", "unsafe", "safe", source="aegis2_test",
                 pert="leetspeak", prefix="rob", latency=40, cost=0.001),
    ]
    full = [
        _receipt("full", "1", "safe", "safe", latency=10, cost=0.001),
        _receipt("full", "2", "unsafe", "unsafe", latency=30, cost=0.003),
        _receipt("full", "L", "unsafe", "unsafe", source="aegis2_test",
                 pert="leetspeak", prefix="rob", latency=80, cost=0.002),
    ]
    frontier = [
        _receipt("frontier-gpt", "1", "safe", "unsafe", latency=100, cost=0.01),
        _receipt("frontier-gpt", "2", "unsafe", "safe", latency=300, cost=0.01),
    ]
    return _write_run(
        tmp,
        {"specialist-only": specialist, "full": full, "frontier-gpt": frontier},
        banner="MOCK run — synthetic unit fixture",
        modes={"specialist-only": "mock", "full": "mock", "frontier-gpt": "mock"},
    )


def test_published_frontier_cells_come_from_benchmarks():
    cells = mr.load_published(ROOT / "data" / "BENCHMARKS.md", ROOT / "data" / "frozen" / "README.md")
    assert cells["OpenAI Moderation"].startswith("PUBLISHED GPT-5.1 macro-F1 80.24")
    assert "unverified" in cells["OpenAI Moderation"]
    assert "not our run" in cells["OpenAI Moderation"]
    assert cells["WildGuardTest"].startswith("PUBLISHED GPT-4o+CoT macro-F1 80.13")
    assert "arXiv:2605.31073" in cells["WildGuardTest"]
    assert cells["XSTest over-block"].startswith("PUBLISHED GPT-4 (jun-2023):")
    assert "6.4%" in cells["XSTest over-block"]
    assert "response-level, not macro-F1" in cells["XSTest over-block"]
    assert cells["Disguised"].startswith("PUBLISHED none")
    context = mr.published_context(ROOT / "data" / "BENCHMARKS.md")
    assert "92.9" in context and "90.6" in context
    assert "not our run" in context


def test_missing_benchmark_line_is_unparsed(tmp_path):
    path = tmp_path / "BENCHMARKS.md"
    path.write_text("# empty\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("no frontier here\n", encoding="utf-8")
    cells = mr.load_published(path, readme)
    assert cells["OpenAI Moderation"].startswith("PUBLISHED unparsed")
    assert cells["WildGuardTest"].startswith("PUBLISHED unparsed")
    assert cells["XSTest over-block"].startswith("PUBLISHED unparsed")
    assert "did not state" in cells["Disguised"]


def test_headline_scales_cost_and_splits_disguise(tmp_path):
    run = _fixture(tmp_path / "run")
    out = mr.build_judge_report(run)
    headline = (out / "headline.md").read_text(encoding="utf-8")
    assert "MOCK run — synthetic unit fixture" in headline
    assert "Every metric below is **MOCK**" in headline
    assert "| OpenAI Moderation | specialist-only | 2 | 100.0 | 1.0000 | 0.0000 | 20.0 | 29.0 | 2,000.00 |" in headline
    assert "| Disguised | full | 1 |" in headline
    assert "| Disguised | specialist-only | 1 |" in headline
    assert "| Disguised | frontier-gpt | 0 |" in headline
    assert "| WildGuardTest | specialist-only | 0 |" in headline
    assert "PUBLISHED GPT-5.1 macro-F1 80.24" in headline
    assert "92.9" in headline
    # $ / 1M is 1000x the harness per-1k figure (2.0), not the per-1k number itself.
    assert "2,000.00" in headline
    openai_line = next(line for line in headline.splitlines() if line.startswith("| OpenAI Moderation | specialist-only |"))
    assert openai_line.split("|")[9].strip() == "2,000.00"


def test_charts_colors_banner_and_offline_index(tmp_path):
    run = _fixture(tmp_path / "run")
    out = mr.build_judge_report(run)
    trade = (out / "tradeoff.svg").read_text(encoding="utf-8")
    assert "MOCK" in trade
    assert "(log scale)" in trade
    assert 'data-role="nvidia"' in trade
    assert 'fill="#76B900"' in trade
    assert 'data-role="frontier"' in trade and 'fill="#767676"' in trade
    spec_line = next(line for line in trade.splitlines()
                     if 'data-system="specialist-only"' in line and "data-usd-per-1m" in line)
    # frontier-gpt ran only omod-1/omod-2, so the tradeoff is scored on those matched rows:
    # 0.004 / 2 rows → $2,000 per 1M (not the 3-row $1,666.67, which would mix row sets).
    assert 'data-role="nvidia"' in spec_line
    assert 'fill="#76B900"' in spec_line
    assert 'data-usd-per-1m="2000.0000"' in spec_line
    assert 'data-n="2"' in spec_line
    assert "Matched 2 rows" in trade
    frontier_line = next(line for line in trade.splitlines()
                         if 'data-system="frontier-gpt"' in line and "data-usd-per-1m" in line)
    assert 'data-usd-per-1m="10000.0000"' in frontier_line

    disguise = (out / "disguise.svg").read_text(encoding="utf-8")
    assert 'data-system="full" data-disguise="leetspeak" data-recall="1.0000"' in disguise
    assert 'data-system="specialist-only" data-disguise="leetspeak" data-recall="0.0000"' in disguise
    assert 'fill="#76B900"' in disguise

    ablation = (out / "ablation.svg").read_text(encoding="utf-8")
    assert 'data-slot="specialist-only" data-measured="yes"' in ablation
    assert 'data-slot="+rules" data-measured="no"' in ablation
    assert 'data-slot="+router" data-measured="no"' in ablation
    assert 'data-slot="+escalation (full)" data-measured="yes"' in ablation
    assert 'data-system="full"' in ablation
    # Gap between specialist-only and full is not drawn as a measured step.
    assert ablation.count("<polyline") == 0

    page = (out / "index.html").read_text(encoding="utf-8")
    assert 'class="banner mock"' in page
    assert "MOCK run — synthetic unit fixture" in page
    assert page.count("<svg") == 3
    assert "<link " not in page
    assert "fonts.googleapis" not in page
    assert "https://" not in page
    assert 'fill="#76B900"' in page


def test_live_banner(tmp_path):
    run = _fixture(tmp_path / "live")
    info = json.loads((run / "RUN_INFO.json").read_text(encoding="utf-8"))
    info["banner"] = "LIVE run — unit fixture"
    info["modes"] = {name: "live" for name in info["modes"]}
    (run / "RUN_INFO.json").write_text(json.dumps(info), encoding="utf-8")
    out = mr.build_judge_report(run)
    page = (out / "index.html").read_text(encoding="utf-8")
    headline = (out / "headline.md").read_text(encoding="utf-8")
    assert 'class="banner live"' in page
    assert ">LIVE<" in page
    assert "LIVE run — unit fixture" in headline
    assert "Metrics below are **LIVE**" in headline
    assert "MOCK" not in (out / "tradeoff.svg").read_text(encoding="utf-8")


def test_summary_only_fallback(tmp_path):
    run = tmp_path / "summary-only"
    run.mkdir()
    summary = {
        "banner": "MOCK run — summary only",
        "systems": {
            "specialist-only": {
                "n": 4,
                "macro_f1": 0.5,
                "unsafe_recall": 0.25,
                "over_block_rate": 0.0,
                "latency_p50_ms": 10,
                "latency_p95_ms": 40,
                "cost_per_1k_usd": 1.5,
                "by_source": {
                    "openai_moderation": {
                        "n": 4, "macro_f1": 0.5, "unsafe_recall": 0.25, "over_block_rate": 0.0,
                        "latency_p50_ms": 10, "latency_p95_ms": 40, "cost_per_1k_usd": 1.5,
                    }
                },
                "by_perturbation": {
                    "leetspeak": {
                        "n": 2, "unsafe_recall": 0.5, "macro_f1": 0.4,
                    }
                },
            }
        },
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    out = mr.build_judge_report(run, benchmarks_path=ROOT / "data" / "BENCHMARKS.md",
                                frozen_readme=ROOT / "data" / "frozen" / "README.md")
    headline = (out / "headline.md").read_text(encoding="utf-8")
    assert "MOCK run — summary only" in headline
    assert "Receipt JSONL was not in this run directory" in headline
    assert "| OpenAI Moderation | specialist-only | 4 | 50.0 | 0.2500 | 0.0000 |" in headline
    assert "| Disguised | specialist-only | 0 |" in headline
    # 1.5 $/1k -> 1,500.00 $/1M
    assert "1,500.00" in headline
    disguise = (out / "disguise.svg").read_text(encoding="utf-8")
    assert 'data-disguise="leetspeak" data-recall="0.5000"' in disguise


def test_mock_integration_e2e_report():
    if not MOCK_RUN.is_dir():
        pytest.skip("MOCK-integration-e2e results are not in this checkout")
    out = mr.build_judge_report(MOCK_RUN)
    info = json.loads((MOCK_RUN / "RUN_INFO.json").read_text(encoding="utf-8"))
    headline = (out / "headline.md").read_text(encoding="utf-8")
    page = (out / "index.html").read_text(encoding="utf-8")
    assert info["banner"] in headline
    assert info["banner"] in page
    assert 'class="banner mock"' in page
    assert "Every metric below is **MOCK**" in headline
    for name in ("OpenAI Moderation", "WildGuardTest", "XSTest over-block", "Disguised"):
        assert name in headline
    assert "PUBLISHED GPT-5.1 macro-F1 80.24" in headline
    assert "PUBLISHED GPT-4o+CoT macro-F1 80.13" in headline
    assert "6.4%" in headline
    assert "PUBLISHED none" in headline
    receipts = mr.load_run(MOCK_RUN)[2]
    public = [r for r in receipts["full"] if r.get("source") == "openai_moderation" and not mr.is_disguise_row(r)]
    disguised = [r for r in receipts["full"] if mr.is_disguise_row(r)]
    wild = [r for r in receipts["full"] if r.get("source") == "wildguard_test" and not mr.is_disguise_row(r)]
    assert f"| OpenAI Moderation | full | {len(public)} |" in headline
    assert f"| Disguised | full | {len(disguised)} |" in headline
    assert wild == []
    assert "| WildGuardTest | full | 0 |" in headline
    trade = (out / "tradeoff.svg").read_text(encoding="utf-8")
    assert "MOCK" in trade and "(log scale)" in trade
    assert 'fill="#76B900"' in trade and 'data-role="frontier"' in trade
    assert 'fill="#767676"' in trade
    disguise = (out / "disguise.svg").read_text(encoding="utf-8")
    for pert in ("none", "leetspeak", "base64_wrap", "homoglyph"):
        assert f'data-disguise="{pert}"' in disguise
    ablation = (out / "ablation.svg").read_text(encoding="utf-8")
    assert 'data-slot="+rules" data-measured="no"' in ablation
    assert 'data-slot="+escalation (full)" data-measured="yes"' in ablation
    assert page.count("<svg") == 3
    assert "<link " not in page


def _subset_run(tmp: Path) -> Path:
    """Ours on 4 public + 2 disguise rows; a frontier arm on 1 public + 2 disguise rows (one refusal)."""
    dep = "accounts/x/deployments/abc"
    ours, front = [], []
    for i, (gold, pred) in enumerate([("unsafe", "unsafe"), ("safe", "safe"), ("safe", "unsafe"), ("unsafe", "unsafe")]):
        row = _receipt("full", str(i), gold, pred, latency=100, cost=0.0)
        row["prediction"]["model"] = dep
        ours.append(row)
    for pert, gold in (("leetspeak", "unsafe"), ("base64_wrap", "unsafe")):
        row = _receipt("full", pert, gold, "unsafe", source="aegis2_test", pert=pert, prefix="rob", latency=100, cost=0.0)
        row["prediction"]["model"] = dep
        ours.append(row)
    ok = _receipt("frontier-router-b", "0", "unsafe", "safe", latency=2000, cost=0.004)
    ok["prediction"]["model"] = "anthropic/claude-opus-5.5"
    leet = _receipt("frontier-router-b", "leetspeak", "unsafe", "unsafe", source="aegis2_test",
                    pert="leetspeak", prefix="rob", latency=3000, cost=0.004)
    leet["prediction"]["model"] = "anthropic/claude-opus-5.5"
    refused = _receipt("frontier-router-b", "base64_wrap", "unsafe", "", source="aegis2_test",
                       pert="base64_wrap", prefix="rob", latency=0, cost=0.0)
    refused["prediction"].update(model="frontier-router-b", route="error")
    refused["error"] = "FrontierContentFilter: anthropic/claude-opus-5.5 finish_reason=content_filter"
    front = [ok, leet, refused]
    run = _write_run(tmp / "run", {"full": ours, "frontier-router-b": front},
                     banner="LIVE run — subset fixture", modes={"full": "live"})
    (run / "nano_summary.json").write_text(json.dumps({"gpu_per_call_usd": 0.0001}), encoding="utf-8")
    (run / "frontier_summary.json").write_text(json.dumps({"label": "LIVE", "arms": {"frontier-router-b": {}}}),
                                               encoding="utf-8")
    ledger = {"entries": [{"provider": "frontier-router", "cost_usd": 0.006,
                           "metadata": {"model": "anthropic/claude-opus-5.5", "row_id": "rob-base64_wrap"}}]}
    (tmp / "spend.json").write_text(json.dumps(ledger), encoding="utf-8")
    return run


def test_subset_systems_get_matched_table_not_mixed_columns(tmp_path):
    run = _subset_run(tmp_path)
    out = mr.build_judge_report(run)
    headline = (out / "headline.md").read_text(encoding="utf-8")
    # All-rows table never puts the 1-row frontier slice next to ours' 4 rows.
    assert "| OpenAI Moderation | frontier-router-b | 1 of 4 — subset, see matched-rows table | n/a |" in headline
    assert "| OpenAI Moderation | full | 4 |" in headline
    # Disguise rows are fully covered by both, so they stay comparable.
    assert "| Disguised | frontier-router-b | 2 |" in headline
    assert "## Matched rows (3 rows every compared system ran)" in headline
    matched = [line for line in headline.splitlines() if line.startswith("| ALL matched rows |")]
    assert len(matched) == 2 and all("| 3 |" in line for line in matched)
    front = next(line for line in matched if "frontier-router-b" in line)
    cells = [c.strip() for c in front.split("|")]
    # Refusal counted unsafe: gold unsafe x3, predicted unsafe x2 -> recall 66.7 %; refusal 1/3.
    assert cells[6] == "66.7" and cells[8] == "33.3"
    # Latency over verdict rows only (2000, 3000) -> p50 2,500, not dragged down by the 0 ms refusal.
    assert cells[9] == "2,500"
    # $ = 0.008 receipts + 0.006 billed refusal from ../spend.json over 3 rows -> $4,667 / 1M.
    assert cells[11] == "4,667"
    ours = next(line for line in matched if "| full |" in line)
    # 3 deployment calls x $0.0001 allocated / 3 rows -> $100 / 1M.
    assert [c.strip() for c in ours.split("|")][11] == "100"
    assert "Claude Opus 5.5" in headline
    assert "frontier-router-b = LIVE (frontier_summary.json)" in headline
    assert "Every metric below" not in headline and "Metrics below are **LIVE**" in headline


def test_subset_charts_use_matched_rows_and_show_refusals(tmp_path):
    run = _subset_run(tmp_path)
    out = mr.build_judge_report(run)
    trade = (out / "tradeoff.svg").read_text(encoding="utf-8")
    front = next(line for line in trade.splitlines() if 'data-system="frontier-router-b"' in line and "<circle" in line)
    assert 'data-n="3"' in front and 'data-refusal="0.3333"' in front
    assert "refused 33.3% (counted unsafe)" in trade
    disguise = (out / "disguise.svg").read_text(encoding="utf-8")
    assert ('data-system="frontier-router-b" data-disguise="base64_wrap" data-recall="1.0000" '
            'data-refusal="1.0000"') in disguise
    assert 'data-refusal-part="frontier-router-b"' in disguise
    assert "100%R" in disguise


def test_ablation_calibrated_and_baseline_reference(tmp_path):
    rows = {
        name: [_receipt(name, "1", "safe", "safe"), _receipt(name, "2", "unsafe", pred)]
        for name, pred in (("specialist-only", "safe"), ("full", "unsafe"), ("full-cal", "unsafe"),
                           ("nvidia-ultra-zeroshot", "unsafe"))
    }
    run = _write_run(tmp_path / "run", rows, banner="MOCK run — ablation fixture",
                     modes={name: "mock" for name in rows})
    out = mr.build_judge_report(run)
    ablation = (out / "ablation.svg").read_text(encoding="utf-8")
    assert 'data-slot="full-cal" data-measured="yes"' in ablation
    assert 'data-baseline="nvidia-ultra-zeroshot"' in ablation
    assert "calibrated variants" in ablation
    assert ablation.count("<polyline") == 0  # specialist-only and full are not adjacent slots


def _vision_dir(tmp: Path) -> Path:
    vdir = tmp / "vision"
    text = [
        _receipt("vision-text-only", "a", "unsafe", "safe", source="siuo", prefix="siuo"),
        _receipt("vision-text-only", "b", "safe", "safe", source="hatemojicheck-rendered", prefix="hm"),
    ]
    routed = [
        _receipt("vision-routed", "a", "unsafe", "unsafe", source="siuo", prefix="siuo"),
        _receipt("vision-routed", "b", "safe", "unsafe", source="hatemojicheck-rendered", prefix="hm"),
    ]
    _write_run(vdir, {"vision-text-only": text, "vision-routed": routed},
               banner="LIVE vision A/B — fixture", modes={"vision-text-only": "live", "vision-routed": "live"})
    (vdir / "summary.json").write_text(json.dumps({"paired": {"unsafe_recall": {
        "n_paired": 2, "difference": 0.5, "ci95": [0.1, 0.9], "mcnemar_p": 0.5}}}), encoding="utf-8")
    return vdir


def test_vision_panel_and_chart_export(tmp_path, monkeypatch):
    run = _fixture(tmp_path / "run")
    vdir = _vision_dir(tmp_path)
    monkeypatch.setattr(mr, "svg_to_png", lambda svg, png, size: False)
    out = mr.build_judge_report(run, vision_dir=vdir, png_dir=tmp_path / "img")
    page = (out / "index.html").read_text(encoding="utf-8")
    assert page.count("<svg") == 4
    vision = (out / "vision.svg").read_text(encoding="utf-8")
    assert 'data-vision-arm="vision-routed" data-source="siuo" data-recall="1.0000"' in vision
    assert 'data-vision-arm="vision-text-only" data-source="siuo" data-recall="0.0000"' in vision
    assert "+50.0 pts" in vision and "LIVE vision A/B — fixture" in vision
    headline = (out / "headline.md").read_text(encoding="utf-8")
    assert "## Image route (LIVE)" in headline
    exported = sorted(p.name for p in (tmp_path / "img").iterdir())
    assert exported == ["run-ablation.svg", "run-disguise.svg", "run-tradeoff.svg", "run-vision.svg"]


def test_frozen_v1_live_report_if_present(tmp_path):
    run = ROOT / "results" / "frozen-v1"
    if not (run / "frontier-router-a.jsonl").is_file():
        pytest.skip("frozen-v1 receipts are not in this checkout")
    _info, _summary, receipts = mr.load_run(run)
    ctx = mr.load_context(run, _info, receipts)
    cov = mr.coverage(mr.system_order(list(receipts)), receipts)
    assert "frontier-router-a" in cov.partial and "full" in cov.full
    rows = mr.restrict(receipts["frontier-router-b"], cov.matched_ids)
    m = mr.score(rows, ctx, ctx.models["frontier-router-b"])
    # FRONTIER_SUMMARY: Opus as-unsafe macro-F1 85.0, refusals 21.6 %, $5,504 / 1M.
    assert round(m["macro_f1"] * 100, 1) == 85.0
    assert round(m["refusal_rate"] * 100, 1) == 21.6
    assert round(mr.per_million(m["cost_per_1k_usd"])) == 5504
