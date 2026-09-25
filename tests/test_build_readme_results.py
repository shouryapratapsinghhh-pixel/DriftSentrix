import os

import pandas as pd


def test_build_section_runs_and_produces_markers(tmp_path, monkeypatch):
    """Doesn't touch the real README/reports -- builds a throwaway
    results.csv + pa_inflation.csv in a tmp dir and confirms the section
    builder produces valid markdown between the expected markers.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("reports", exist_ok=True)

    pd.DataFrame(
        [
            {"model": "zscore", "threshold_mode": "oracle", "point_f1": 1.0, "pa_f1": 1.0, "event_f1": 1.0,
             "false_alarms_per_1000": 0.0},
            {"model": "zscore", "threshold_mode": "train_percentile_p99.5", "point_f1": 0.9, "pa_f1": 0.9,
             "event_f1": 0.8, "false_alarms_per_1000": 2.0},
        ]
    ).to_csv("reports/results.csv", index=False)

    pd.DataFrame(
        [{"model": "zscore", "random_point_f1": 0.1, "random_pa_f1": 0.6}]
    ).to_csv("reports/pa_inflation.csv", index=False)

    # Import scripts/build_readme_results.py directly by path (scripts/ isn't a package)
    import importlib.util

    this_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(this_dir, "..", "scripts", "build_readme_results.py")
    spec = importlib.util.spec_from_file_location("build_readme_results", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    section = module.build_section()
    assert "<!-- RESULTS:START -->" in section
    assert "<!-- RESULTS:END -->" in section
    assert "zscore" in section
    assert "PA-inflation" in section


def test_main_updates_readme_between_markers(tmp_path):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(this_dir, "..", "scripts", "build_readme_results.py")

    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\n<!-- RESULTS:START -->\nold content\n<!-- RESULTS:END -->\n\nfooter\n")

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    pd.DataFrame(
        [{"model": "zscore", "threshold_mode": "oracle", "point_f1": 1.0, "pa_f1": 1.0, "event_f1": 1.0,
          "false_alarms_per_1000": 0.0}]
    ).to_csv(reports_dir / "results.csv", index=False)

    import importlib.util
    import os as _os

    old_cwd = _os.getcwd()
    _os.chdir(tmp_path)
    try:
        spec = importlib.util.spec_from_file_location("build_readme_results", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    finally:
        _os.chdir(old_cwd)

    updated = readme.read_text()
    assert "old content" not in updated
    assert "footer" in updated  # content after the marker is preserved
    assert "# Title" in updated  # content before the marker is preserved
