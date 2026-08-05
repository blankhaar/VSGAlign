from paper.plot_single_grain_overview import main as plot_single_grain_overview


def test_single_grain_paper_script_smoke(tmp_path):
    outputs = plot_single_grain_overview(output_dir=tmp_path, n_nu=30, sigma=0.1)
    assert outputs
    for output in outputs:
        assert output.exists()
