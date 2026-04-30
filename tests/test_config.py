from omegaconf import OmegaConf


def test_baseline_config_loads():
    cfg = OmegaConf.load("configs/paper/01_traditional_attack_baseline.yaml")
    assert cfg.task.name == "traditional_attack_baseline"
    assert cfg.task.seed == 42
    assert len(cfg.attacks) == 2
    assert len(cfg.defenses) == 3


def test_diffusion_config_loads():
    cfg = OmegaConf.load("configs/paper/02_generative_attack_mainline.yaml")
    assert cfg.task.name == "generative_attack_mainline"
    assert len(cfg.attacks) == 1
    assert cfg.attacks[0].name == "diffusion"
    assert "lpips" in cfg.metrics.quality


def test_full_suite_config_loads():
    cfg = OmegaConf.load("configs/paper/03_full_resnet50_suite.yaml")
    assert cfg.task.name == "full_resnet50_suite"
    assert [a.name for a in cfg.attacks] == ["fgsm", "pgd", "diffusion"]
    assert [d.name for d in cfg.defenses] == [
        "jpeg",
        "gaussian_blur",
        "bit_depth",
        "diffusion_purification",
    ]
    assert "fid" in cfg.metrics.quality


def test_ablation_configs_load():
    paths = [
        "configs/ablations/defense/defense_00_no_defense.yaml",
        "configs/ablations/diffusion_strength/diffusion_strength_03.yaml",
        "configs/ablations/purification_steps/purification_steps_10.yaml",
    ]
    for path in paths:
        cfg = OmegaConf.load(path)
        assert cfg.task.name
