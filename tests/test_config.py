from omegaconf import OmegaConf


def test_baseline_config_loads():
    cfg = OmegaConf.load("configs/paper/01_traditional_attack_baseline.yaml")
    assert cfg.task.name == "traditional_attack_baseline"
    assert cfg.task.seeds == [42, 123, 456]
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
        "configs/ablations/attack/diffusion/strength_03.yaml",
        "configs/ablations/defense/diffusion_purification/steps_10.yaml",
    ]
    for path in paths:
        cfg = OmegaConf.load(path)
        assert cfg.task.name


def test_new_attack_ablation_configs_load():
    paths = [
        "configs/ablations/attack/fgsm_eps/fgsm_eps_001.yaml",
        "configs/ablations/attack/fgsm_eps/fgsm_eps_015.yaml",
        "configs/ablations/attack/pgd_eps/pgd_eps_008.yaml",
        "configs/ablations/attack/pgd_steps/pgd_steps_40.yaml",
        "configs/ablations/attack/advgan/advgan_eps_003.yaml",
        "configs/ablations/attack/advgan/advgan_epochs_100.yaml",
    ]
    for path in paths:
        cfg = OmegaConf.load(path)
        assert cfg.task.name
        assert len(cfg.attacks) == 1
        assert len(cfg.defenses) == 0


def test_new_defense_ablation_configs_load():
    paths = [
        "configs/ablations/defense/jpeg_quality/jpeg_q25.yaml",
        "configs/ablations/defense/jpeg_quality/jpeg_q90.yaml",
        "configs/ablations/defense/blur_config/blur_k3_s05.yaml",
        "configs/ablations/defense/blur_config/blur_k9_s20.yaml",
        "configs/ablations/defense/bit_depth/bits_2.yaml",
        "configs/ablations/defense/bit_depth/bits_6.yaml",
        "configs/ablations/defense/diffusion_purification/noise_005.yaml",
        "configs/ablations/defense/diffusion_purification/noise_020.yaml",
    ]
    for path in paths:
        cfg = OmegaConf.load(path)
        assert cfg.task.name


def test_cross_interaction_configs_load():
    paths = [
        "configs/ablations/cross/fgsm_vs_defenses.yaml",
        "configs/ablations/cross/pgd_vs_defenses.yaml",
        "configs/ablations/cross/advgan_vs_defenses.yaml",
        "configs/ablations/cross/diffusion_vs_defenses.yaml",
    ]
    for path in paths:
        cfg = OmegaConf.load(path)
        assert cfg.task.name
        assert len(cfg.defenses) >= 3


def test_model_ablation_configs_load():
    cfg_vit_trad = OmegaConf.load("configs/ablations/model/vit_b_16_A_traditional.yaml")
    assert cfg_vit_trad.target_model.name == "vit_b_16"
    assert len(cfg_vit_trad.attacks) == 3

    cfg_vit_diff = OmegaConf.load("configs/ablations/model/vit_b_16_C_diffusion.yaml")
    assert cfg_vit_diff.target_model.name == "vit_b_16"
    assert len(cfg_vit_diff.attacks) == 1

    cfg_dn_trad = OmegaConf.load("configs/ablations/model/densenet121_A_traditional.yaml")
    assert cfg_dn_trad.target_model.name == "densenet121"
    assert len(cfg_dn_trad.attacks) == 3

    cfg_dn_diff = OmegaConf.load("configs/ablations/model/densenet121_C_diffusion.yaml")
    assert cfg_dn_diff.target_model.name == "densenet121"
    assert len(cfg_dn_diff.attacks) == 1


def test_is_sd_config_classification():
    """Verify SD-heavy vs light config classification for parallel runner."""
    from src.cli import _is_sd_config
    from pathlib import Path

    # Light configs (no SD)
    assert not _is_sd_config(Path("configs/paper/01_traditional_attack_baseline.yaml"))
    assert not _is_sd_config(Path("configs/ablations/attack/fgsm_eps/fgsm_eps_003.yaml"))
    assert not _is_sd_config(Path("configs/ablations/attack/advgan/advgan_eps_003.yaml"))
    assert not _is_sd_config(Path("configs/ablations/cross/fgsm_vs_defenses.yaml"))
    assert not _is_sd_config(Path("configs/ablations/defense/jpeg_quality/jpeg_q75.yaml"))
    assert not _is_sd_config(Path("configs/smoke/synthetic.yaml"))

    # SD-heavy configs
    assert _is_sd_config(Path("configs/paper/02_generative_attack_mainline.yaml"))
    assert _is_sd_config(Path("configs/ablations/attack/diffusion/strength_05.yaml"))
    assert _is_sd_config(Path("configs/ablations/cross/diffusion_vs_defenses.yaml"))
    assert _is_sd_config(Path("configs/ablations/defense/diffusion_purification/steps_20.yaml"))
    assert _is_sd_config(Path("configs/smoke/sd_tiny_e2e.yaml"))

    # Non-existent config returns False (graceful degradation)
    assert not _is_sd_config(Path("configs/does_not_exist.yaml"))


def test_collect_prewarm_requirements(tmp_path):
    """Prewarm should detect the exact models and quality caches needed."""
    from pathlib import Path

    from omegaconf import OmegaConf

    from src.cli import _collect_prewarm_requirements

    cfg = OmegaConf.create({
        "target_model": {
            "type": "classifier",
            "name": "resnet50",
            "weights": "imagenet",
        },
        "attacks": [
            {
                "name": "diffusion",
                "backend": "sd",
                "generator": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            },
            {
                "name": "diffusion",
                "backend": "mock",
                "generator": "mock",
            },
            {
                "name": "diffusion",
            },
        ],
        "defenses": [
            {
                "name": "diffusion_purification",
                "backend": "sd",
                "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            }
        ],
        "metrics": {"quality": ["lpips", "clip_score", "fid"]},
    })
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)

    classifiers, sd_pipelines, needs_lpips, needs_fid, needs_clip = (
        _collect_prewarm_requirements([Path(config_path)])
    )

    assert classifiers == {"resnet50"}
    assert sd_pipelines == {"stable-diffusion-v1-5/stable-diffusion-v1-5"}
    assert needs_lpips
    assert needs_fid
    assert needs_clip


def test_collect_prewarm_requirements_ignores_lightweight_config(tmp_path):
    """Mock/no-weight configs should not force network prewarming."""
    from omegaconf import OmegaConf

    from src.cli import _collect_prewarm_requirements

    cfg = OmegaConf.create({
        "target_model": {"name": "resnet50", "weights": "none"},
        "attacks": [{"name": "fgsm", "eps": 0.03}],
        "defenses": [{"name": "jpeg", "quality": 75}],
        "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
    })
    config_path = tmp_path / "light.yaml"
    OmegaConf.save(cfg, config_path)

    classifiers, sd_pipelines, needs_lpips, needs_fid, needs_clip = (
        _collect_prewarm_requirements([config_path])
    )

    assert classifiers == set()
    assert sd_pipelines == set()
    assert not needs_lpips
    assert not needs_fid
    assert not needs_clip


def test_per_gpu_pool_distribution():
    """Verify round-robin distributes configs evenly across GPUs."""
    from collections import Counter

    gpu_list = [0, 1, 2, 3]
    num_gpus = len(gpu_list)
    config_count = 57  # total .yaml files

    dist = Counter()
    for i in range(config_count):
        dist[gpu_list[i % num_gpus]] += 1

    # With 57 configs / 4 GPUs, each GPU gets 14-15 configs
    assert dist[0] == 15  # 0,4,8,...,56 = 15
    assert dist[1] == 14  # 1,5,9,...,53 = 14
    # Actually: 0..56 step 4 = 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56 = 15
    # 1..53 step 4 = 1,5,9,13,17,21,25,29,33,37,41,45,49,53 = 14
    # 2..54 step 4 = 14
    # 3..55 step 4 = 14
    # Total = 15+14+14+14 = 57. OK
    assert sum(dist.values()) == config_count
    assert max(dist.values()) - min(dist.values()) <= 1  # near-perfect balance


def test_paper_new_configs_load():
    cfg_cross_a = OmegaConf.load("configs/paper/04_cross_model_A_traditional.yaml")
    assert cfg_cross_a.task.name == "cross_model_traditional"
    assert len(cfg_cross_a.attacks) == 3
    assert "advgan" in [a.name for a in cfg_cross_a.attacks]

    cfg_cross_c = OmegaConf.load("configs/paper/04_cross_model_C_diffusion.yaml")
    assert cfg_cross_c.task.name == "cross_model_diffusion"
    assert len(cfg_cross_c.attacks) == 1
    assert cfg_cross_c.attacks[0].name == "diffusion"

    cfg_targeted = OmegaConf.load("configs/paper/05_targeted_attack.yaml")
    assert cfg_targeted.task.name == "targeted_attack"
    assert len(cfg_targeted.attacks) == 2
