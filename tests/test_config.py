from omegaconf import OmegaConf


def test_baseline_config_loads():
    cfg = OmegaConf.load("configs/baseline_resnet50.yaml")
    assert cfg.task.name == "baseline_resnet50"
    assert cfg.task.seed == 42
    assert len(cfg.attacks) == 2
    assert len(cfg.defenses) == 3
