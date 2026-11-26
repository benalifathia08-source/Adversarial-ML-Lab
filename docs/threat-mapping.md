# Threat mapping: MITRE ATLAS and OWASP Machine Learning Security Top 10

Technique and mitigation names are from the official MITRE ATLAS data file, version 2026.07.

## Attacks tested

| What this lab does | MITRE ATLAS technique | OWASP ML Top 10 (2023) | Code |
|---|---|---|---|
| FGSM / PGD on the feature vector (white-box) | AML.T0043.000 Craft Adversarial Data: White-Box Optimization; AML.T0015 Evade AI Model | ML01 Input Manipulation Attack | `attacks.art_fgsm`, `attacks.art_pgd` |
| HopSkipJump and random search with decisions only (black-box) | AML.T0043.001 Black-Box Optimization; AML.T0015 | ML01 | `attacks.art_hopskipjump`, `attacks.random_increase` |
| Adversarial flows crafted on the MLP, replayed against LightGBM | AML.T0043.002 Black-Box Transfer | ML01 | `stages/evasion.py` (`transfer_to_lgbm`) |
| Constrained evasion: only add traffic, keep the flow consistent | AML.T0043.000 + AML.T0015, in problem space | ML01 | `attacks.constrained_pgd` |
| Label flipping (random and targeted at one attack family) | AML.T0020 Training Data Poisoning | ML02 Data Poisoning Attack | `poison.flip_labels` |
| Backdoor: triggered attack flows labelled normal | AML.T0020 → AML.T0018.000 Poison AI Model | ML02, ML10 Model Poisoning | `poison.inject_backdoor` |
| Pickle checkpoint that runs code on load | AML.T0010.003 AI Supply Chain Compromise: Model; AML.T0011.000 User Execution: Unsafe AI Artifacts | ML06 AI Supply Chain Attacks | `supply_chain.make_malicious_checkpoint` |
| Tampering with weights, preprocessing or the manifest | AML.T0018 Manipulate AI Model; AML.T0010.003 | ML06, ML10 | `stages/supply_chain.py` (tamper tests) |

## Defenses tested

| Defense | MITRE ATLAS mitigation | Result in this lab |
|---|---|---|
| Input validation (one-hot, integers, bytes/packets arithmetic) | AML.M0015 Predictive AI Adversarial Input Detection | Rejects every FGSM/PGD output and 77% of HopSkipJump outputs; useless against the constrained attack, by design |
| Adversarial training with the constrained attack | AML.M0003 Predictive AI Model Hardening | Evasion 61% → 0.1%, false positives 12% → 17% |
| Robust features (drop what the attacker controls) | AML.M0003 | Evasion 0% under both threat models, false positives 11% (released) |
| Ensemble (MLP OR LightGBM) | AML.M0003 | Does not help: evasions transfer (59%) |
| Trusted core set + frequency-spike check | AML.M0007 Sanitize Training Data; AML.M0025 Maintain AI Dataset Provenance | Removes 100% of a 1% backdoor; backdoor 90% → 1.7% (trigger alone: 1.3%) |
| ART activation clustering, spectral signatures, cross-validated label check | AML.M0007 | Fail on this tabular data (kept in the results on purpose) |
| safetensors + JSON only, `torch.load(weights_only=True)` | AML.M0011 Restrict Library Loading | Payload never runs |
| ModelScan in the gate and in CI | AML.M0016 Vulnerability Scanning | Flags the malicious checkpoint (CRITICAL) |
| Ed25519-signed manifest, verified before every load | AML.M0013 Code Signing; AML.M0014 Verify AI Artifacts | One flipped bit, an edited file, an extra file, a missing signature or another key: all refused |
| Promotion gate (signature + scan + accuracy + robustness) and MLflow registry alias | AML.M0008 Validate AI Model; AML.M0017 AI Model Distribution Methods | Only the gate-approved model (robust features) is registered as `champion`; baseline and adversarially trained models are rejected |
| Pinned dataset hashes, model manifest lists dataset hashes and git commit | AML.M0025; AML.M0023 AI Bill of Materials | Wrong dataset file is refused |

## What this lab does not cover

- An attacker who also shapes the sensor's connection counters (`ct_*`: opening many extra connections). They are computed
  over time windows by the sensor; modelling them needs a traffic replay, not feature edits.
- Model extraction, membership inference and model inversion (privacy attacks).
- Certified robustness (e.g. randomized smoothing).
