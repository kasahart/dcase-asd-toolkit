from pathlib import Path


def test_ap_only_rows_use_knn_varmin_without_beam():
    recipe = Path("jobs/asd/recipe/raw_beats_beam_rdp4.sh").read_text()
    global_case = recipe.split("global_ap)", 1)[1].split(";;", 1)[0]
    freq_case = recipe.split("freq_ap)", 1)[1].split(";;", 1)[0]

    for case in [global_case, freq_case]:
        assert 'experiments_score="knn_varmin4"' in case
        assert "beam_" not in case
        assert 'experiments_score="default"' not in case


def test_dcase2026_recipe_preserves_original_duration():
    recipe = Path("jobs/asd/recipe/raw_beats_beam_rdp4.sh").read_text()
    dcase2026 = recipe.split('if [ "${dcase}" = "dcase2026" ]', 1)[1]
    assert '"datamodule.train.collator.sec=all"' in dcase2026
    assert '"datamodule.train.dataloader.batch_size=1"' in dcase2026
