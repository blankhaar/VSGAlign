from validation.continuum_vs_discrete import generate_report


def test_validation_visual_smoke(tmp_path):
    result = generate_report(tmp_path, n_nu=30, jmax=60)
    assert result["plot_path"].exists()
    assert result["metrics_path"].exists()
