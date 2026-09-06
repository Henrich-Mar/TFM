import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v2_self_play import _pretrain_report_allows_ppo


def test_pretrain_report_accepts_utf8_with_bom(tmp_path):
    report = tmp_path / "pretrain_report.json"
    report.write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"ppo_gate_passed": True}).encode("utf-8")
    )

    assert _pretrain_report_allows_ppo(report) is True


def test_pretrain_report_still_accepts_plain_utf8(tmp_path):
    report = tmp_path / "pretrain_report.json"
    report.write_text(json.dumps({"ppo_gate_passed": True}), encoding="utf-8")

    assert _pretrain_report_allows_ppo(report) is True
