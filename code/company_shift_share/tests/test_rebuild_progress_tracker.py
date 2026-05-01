from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from company_shift_share import rebuild_progress_tracker as tracker


class RebuildProgressTrackerTests(unittest.TestCase):
    def test_parse_selected_positions_stage(self) -> None:
        log_text = """
        [company_features] Building source-based firm features for 2010–2015
        [wrds_workforce_cache] BUILD: /tmp/workforce.parquet | 243,425 firms | years 2010–2015
        [wrds_workforce_extract] Proactive large-firm slicing for selected-us-positions: 1,290 firm(s) with n_users >= 75,000 scheduled as 1,290 singleton task(s) using 1-year windows
        [wrds_workforce_extract] DONE  workforce-selected-us-positions batch 534/4865: 12,476 selected-us-position rows | elapsed 0.2s
        [wrds_workforce_extract] START workforce-selected-us-positions batch 535/4865: 50 firms | history years 2010–2015
        """
        status = tracker.parse_rebuild_log(log_text)
        self.assertEqual(status.stage_key, "selected_positions_extract")
        self.assertEqual(status.progress_done, 534)
        self.assertEqual(status.progress_total, 6155)
        self.assertAlmostEqual(status.progress_percent or 0.0, 534 / 6155 * 100.0, places=6)

    def test_parse_user_history_stage_and_eta(self) -> None:
        log_text = """
        [wrds_workforce_extract] DONE selected-us-positions scan: 80 chunks | elapsed 24.7s
        [wrds_workforce_extract] START user-history extract: 1,260,000 user ids | 63 chunks | workers=3
        [wrds_workforce_extract] worker 1/3 import chunk 00001/00063: user_ids=20,000
        [wrds_workforce_extract] worker 2/3 import chunk 00002/00063: user_ids=20,000
        [wrds_workforce_extract] worker 3/3 import chunk 00003/00063: user_ids=20,000
        """
        status = tracker.parse_rebuild_log(log_text)
        self.assertEqual(status.stage_key, "user_history_extract")
        self.assertEqual(status.progress_done, 3)
        self.assertEqual(status.progress_total, 63)

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_path = tmp / "tracker_state.json"
            status_path = tmp / "tracker_status.txt"
            log_path = tmp / "rebuild.log"
            log_path.write_text(log_text)

            first = tracker.track_once(
                log_path=log_path,
                state_path=state_path,
                status_path=status_path,
                subject_prefix="[test]",
            )
            self.assertIsNone(first.stage_eta_seconds)

            log_path.write_text(log_text.replace("00003/00063", "00008/00063"))
            state = tracker.load_state(state_path)
            state["observations"][0]["ts"] = "2026-04-20T00:00:00+00:00"
            tracker.save_state(state_path, state)

            second = tracker.track_once(
                log_path=log_path,
                state_path=state_path,
                status_path=status_path,
                subject_prefix="[test]",
            )
            self.assertEqual(second.stage_key, "user_history_extract")
            self.assertIsNotNone(second.stage_eta_seconds)
            self.assertTrue((status_path.with_suffix(status_path.suffix + ".json")).exists())


if __name__ == "__main__":
    unittest.main()
