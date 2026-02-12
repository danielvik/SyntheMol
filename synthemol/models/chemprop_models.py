"""Contains training and predictions functions for Chemprop models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from synthemol.constants import CHEMPROP_VERSIONS, FEATURES_SIZE_MAPPING, H2O_FEATURES


def _normalize_chemprop_version(chemprop_version: CHEMPROP_VERSIONS | str | None) -> CHEMPROP_VERSIONS:
    if chemprop_version is None:
        return "v2"

    version = str(chemprop_version).lower().strip()
    if version in {"v1", "1", "1.x", "1.6.1"}:
        return "v1"
    if version in {"v2", "2", "2.x", "2.2.2"}:
        return "v2"

    raise ValueError(
        f"Unsupported chemprop_version={chemprop_version!r}. Expected 'v1' or 'v2'."
    )


# ----------------------------
# Chemprop v1 implementations
# ----------------------------

def _chemprop_build_model_v1(
    dataset_type: str,
    features_type: str | None = None,
    property_name: str = "task",
):
    from chemprop.args import TrainArgs
    from chemprop.models import MoleculeModel

    arg_list = [
        "--data_path",
        "foo.csv",
        "--dataset_type",
        dataset_type,
        "--save_dir",
        "foo",
        "--quiet",
    ]

    if features_type == "rdkit":
        arg_list += [
            "--features_generator",
            "rdkit_2d_normalized",
            "--no_features_scaling",
        ]

    if features_type == "morgan":
        arg_list += [
            "--features_generator",
            "morgan",
            "--no_features_scaling",
        ]

    args = TrainArgs().parse_args(arg_list)
    args.task_names = [property_name]

    if features_type is not None:
        args.features_size = FEATURES_SIZE_MAPPING[features_type]

    torch.manual_seed(0)

    return MoleculeModel(args)


def _chemprop_load_v1(model_path: Path, device: torch.device) -> Any:
    from chemprop.utils import load_checkpoint

    return load_checkpoint(path=str(model_path), device=device).eval()


def _chemprop_load_scaler_v1(model_path: Path) -> StandardScaler:
    from chemprop.utils import load_scalers

    return load_scalers(path=str(model_path))[0]


def _chemprop_predict_on_molecule_v1(
    model: Any,
    smiles: str,
    fingerprint: np.ndarray | None = None,
    scaler: StandardScaler | None = None,
    h2o_solvents: bool = False,
) -> float:
    extra_features = H2O_FEATURES if h2o_solvents else []

    pred = model(
        batch=[[smiles]],
        features_batch=[np.concatenate((fingerprint, extra_features), axis=0)]
        if fingerprint is not None
        else None,
    ).item()

    if scaler is not None:
        pred = scaler.inverse_transform([[pred]])[0][0]

    return float(pred)


# ----------------------------
# Chemprop v2 implementations
# ----------------------------

def _chemprop_build_model_v2(
    dataset_type: str,
    features_type: str | None = None,
    property_name: str = "task",
):
    # NOTE: property_name is currently unused in v2 model construction.
    from chemprop import models, nn

    # Message passing and aggregation use Chemprop defaults.
    message_passing = nn.BondMessagePassing()
    aggregation = nn.MeanAggregation()

    # Determine input dimension for the predictor when using extra descriptors.
    input_dim = None
    if features_type is not None:
        features_size = FEATURES_SIZE_MAPPING[features_type]
        # Chemprop v2 defaults to hidden dim = 300; we use that as the base.
        # This keeps behavior consistent with v1 while supporting concatenated descriptors.
        input_dim = 300 + features_size

    if dataset_type == "classification":
        if input_dim is None:
            predictor = nn.BinaryClassificationFFN(n_tasks=1)
        else:
            predictor = nn.BinaryClassificationFFN(n_tasks=1, input_dim=input_dim)
        metrics = None
    elif dataset_type == "regression":
        if input_dim is None:
            predictor = nn.RegressionFFN(n_tasks=1)
        else:
            predictor = nn.RegressionFFN(n_tasks=1, input_dim=input_dim)
        metrics = None
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported.")

    if metrics is None:
        model = models.MPNN(message_passing, aggregation, predictor)
    else:
        model = models.MPNN(message_passing, aggregation, predictor, metrics=metrics)

    return model


def _chemprop_load_v2(model_path: Path, device: torch.device) -> Any:
    from chemprop.models import MPNN

    if model_path.suffix == ".ckpt":
        model = MPNN.load_from_checkpoint(model_path)
    else:
        model = MPNN.load_from_file(model_path)
    model = model.to(device)
    model.eval()
    return model


def _chemprop_predict_on_molecule_v2(
    model: Any,
    smiles: str,
    fingerprint: np.ndarray | None = None,
    h2o_solvents: bool = False,
) -> float:
    from chemprop import data, featurizers

    # Assemble optional descriptor features
    if fingerprint is not None:
        extra_features = H2O_FEATURES if h2o_solvents else []
        x_d = np.concatenate((fingerprint, extra_features), axis=0).astype(np.float32)
    else:
        x_d = None

    datapoint = data.MoleculeDatapoint.from_smi(smiles, y=None, x_d=x_d)
    dataset = data.MoleculeDataset([datapoint], featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer())
    dataloader = data.build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    if hasattr(batch, "bmg"):
        bmg = batch.bmg
        v_d = batch.V_d
        x_d = batch.X_d
    else:
        bmg, v_d, x_d, *_ = batch

    moved = bmg.to(next(model.parameters()).device) if hasattr(bmg, "to") else bmg
    if moved is not None:
        bmg = moved
    if v_d is not None:
        v_d = v_d.to(next(model.parameters()).device)
    if x_d is not None:
        x_d = x_d.to(next(model.parameters()).device)

    with torch.inference_mode():
        preds = model(bmg, v_d, x_d)

    return float(preds.squeeze().item())


# ----------------------------
# Public API (versioned)
# ----------------------------

def chemprop_build_model(
    dataset_type: str,
    features_type: str | None = None,
    property_name: str = "task",
    chemprop_version: CHEMPROP_VERSIONS | str | None = None,
) -> Any:
    """Builds a Chemprop model for v1 or v2."""
    version = _normalize_chemprop_version(chemprop_version)
    if version == "v1":
        return _chemprop_build_model_v1(
            dataset_type=dataset_type,
            features_type=features_type,
            property_name=property_name,
        )

    return _chemprop_build_model_v2(
        dataset_type=dataset_type,
        features_type=features_type,
        property_name=property_name,
    )


def chemprop_load(
    model_path: Path, device: torch.device = torch.device("cpu"), chemprop_version: CHEMPROP_VERSIONS | str | None = None
) -> Any:
    """Loads a Chemprop model for v1 or v2."""
    version = _normalize_chemprop_version(chemprop_version)
    if version == "v1":
        return _chemprop_load_v1(model_path=model_path, device=device)

    return _chemprop_load_v2(model_path=model_path, device=device)


def chemprop_load_scaler(
    model_path: Path, chemprop_version: CHEMPROP_VERSIONS | str | None = None
) -> StandardScaler | None:
    """Loads a Chemprop scaler for v1; returns None for v2."""
    version = _normalize_chemprop_version(chemprop_version)
    if version == "v1":
        return _chemprop_load_scaler_v1(model_path=model_path)

    return None


def chemprop_predict_on_molecule(
    model: Any,
    smiles: str,
    fingerprint: np.ndarray | None = None,
    scaler: StandardScaler | None = None,
    h2o_solvents: bool = False,
    chemprop_version: CHEMPROP_VERSIONS | str | None = None,
) -> float:
    """Predicts a property value for a single molecule."""
    version = _normalize_chemprop_version(chemprop_version)
    if version == "v1":
        return _chemprop_predict_on_molecule_v1(
            model=model,
            smiles=smiles,
            fingerprint=fingerprint,
            scaler=scaler,
            h2o_solvents=h2o_solvents,
        )

    return _chemprop_predict_on_molecule_v2(
        model=model,
        smiles=smiles,
        fingerprint=fingerprint,
        h2o_solvents=h2o_solvents,
    )


def chemprop_predict_on_molecule_ensemble(
    models: list[Any],
    smiles: str,
    fingerprint: np.ndarray | None = None,
    scalers: list[StandardScaler] | None = None,
    h2o_solvents: bool = False,
    chemprop_version: CHEMPROP_VERSIONS | str | None = None,
) -> float:
    """Predicts a property value for a single molecule using an ensemble."""
    version = _normalize_chemprop_version(chemprop_version)
    if version == "v1":
        if scalers is None:
            scalers = [None] * len(models)

        return float(
            np.mean(
                [
                    _chemprop_predict_on_molecule_v1(
                        model=model,
                        smiles=smiles,
                        fingerprint=fingerprint,
                        scaler=scaler,
                        h2o_solvents=h2o_solvents,
                    )
                    for model, scaler in zip(models, scalers)
                ]
            )
        )

    return float(
        np.mean(
            [
                _chemprop_predict_on_molecule_v2(
                    model=model,
                    smiles=smiles,
                    fingerprint=fingerprint,
                    h2o_solvents=h2o_solvents,
                )
                for model in models
            ]
        )
    )
