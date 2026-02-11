"""Contains training and predictions functions for Chemprop models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def _normalize_chemprop_version(chemprop_version: str | None) -> str:
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

def _chemprop_predict_v1(
    model: Any,
    smiles: list[str],
    fingerprints: np.ndarray | None = None,
    num_workers: int = 0,
) -> np.ndarray:
    from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
    from chemprop.train import predict as _chemprop_predict

    if fingerprints is None:
        fingerprints = [None] * len(smiles)

    data_loader = MoleculeDataLoader(
        dataset=MoleculeDataset(
            [
                MoleculeDatapoint(smiles=[smi], targets=None, features=fp)
                for smi, fp in zip(smiles, fingerprints)
            ]
        ),
        num_workers=num_workers,
        shuffle=False,
    )

    preds = np.array(_chemprop_predict(model=model, data_loader=data_loader))[:, 0]
    return preds


def _chemprop_train_v1(
    dataset_type: str,
    train_smiles: list[str],
    val_smiles: list[str],
    fingerprint_type: str | None,
    train_fingerprints: np.ndarray | None,
    val_fingerprints: np.ndarray | None,
    property_name: str,
    train_properties: list[int],
    val_properties: list[int],
    epochs: int,
    save_path: Path,
    num_workers: int = 0,
    use_gpu: bool = False,
) -> Any:
    from chemprop.args import TrainArgs
    from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
    from chemprop.models import MoleculeModel
    from chemprop.train import get_loss_func, train as _chemprop_train
    from chemprop.utils import build_lr_scheduler, build_optimizer, load_checkpoint, save_checkpoint
    from sklearn.metrics import average_precision_score, mean_absolute_error
    from tqdm import trange

    arg_list = [
        "--data_path",
        "foo.csv",
        "--dataset_type",
        dataset_type,
        "--save_dir",
        "foo",
        "--epochs",
        str(epochs),
        "--quiet",
    ] + ([] if use_gpu else ["--no_cuda"])

    if fingerprint_type == "morgan":
        arg_list += ["--features_generator", "morgan"]
    elif fingerprint_type == "rdkit":
        arg_list += [
            "--features_generator",
            "rdkit_2d_normalized",
            "--no_features_scaling",
        ]
    elif fingerprint_type is None:
        pass
    else:
        raise ValueError(f'Fingerprint type "{fingerprint_type}" is not supported.')

    args = TrainArgs().parse_args(arg_list)
    args.task_names = [property_name]
    args.train_data_size = len(train_smiles)

    if fingerprint_type is not None:
        args.features_size = train_fingerprints.shape[1]

    torch.manual_seed(0)

    if not use_gpu:
        torch.use_deterministic_algorithms(True)

    def build_loader(smiles, fingerprints, properties, shuffle):
        if fingerprints is None:
            fingerprints = [None] * len(smiles)
        if properties is None:
            properties = [None] * len(smiles)
        else:
            properties = [[float(prop)] for prop in properties]

        return MoleculeDataLoader(
            dataset=MoleculeDataset(
                [
                    MoleculeDatapoint(smiles=[smi], targets=prop, features=fp)
                    for smi, fp, prop in zip(smiles, fingerprints, properties)
                ]
            ),
            num_workers=num_workers,
            shuffle=shuffle,
        )

    train_loader = build_loader(
        smiles=train_smiles,
        fingerprints=train_fingerprints,
        properties=train_properties,
        shuffle=True,
    )

    model = MoleculeModel(args)

    loss_func = get_loss_func(args)
    optimizer = build_optimizer(model, args)
    scheduler = build_lr_scheduler(optimizer, args)

    save_path = str(save_path)
    best_score = float("inf") if args.minimize_score else -float("inf")
    best_epoch = n_iter = 0
    for epoch in trange(args.epochs):
        n_iter = _chemprop_train(
            model=model,
            data_loader=train_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            n_iter=n_iter,
        )

        val_probs = _chemprop_predict_v1(
            model=model, smiles=val_smiles, fingerprints=val_fingerprints
        )

        if dataset_type == "classification":
            val_score = average_precision_score(val_properties, val_probs)
            new_best_val_score = val_score > best_score
        elif dataset_type == "regression":
            val_score = mean_absolute_error(val_properties, val_probs)
            new_best_val_score = val_score < best_score
        else:
            raise ValueError(f'Dataset type "{dataset_type}" is not supported.')

        if new_best_val_score:
            best_score, best_epoch = val_score, epoch
            save_checkpoint(path=save_path, model=model, args=args)

    model = load_checkpoint(save_path, device=args.device)
    return model


# ----------------------------
# Chemprop v2 implementations
# ----------------------------

def _chemprop_predict_v2(
    model: Any,
    smiles: list[str],
    fingerprints: np.ndarray | None = None,
    num_workers: int = 0,
) -> np.ndarray:
    from chemprop import data, featurizers

    x_d_list = None
    if fingerprints is not None:
        x_d_list = [fp.astype(np.float32) for fp in fingerprints]

    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y=None, x_d=x_d)
        for smi, x_d in zip(smiles, x_d_list or [None] * len(smiles))
    ]

    dataset = data.MoleculeDataset(
        datapoints, featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer()
    )

    dataloader = data.build_dataloader(
        dataset, batch_size=min(50, len(smiles)), shuffle=False, num_workers=num_workers
    )

    device = next(model.parameters()).device
    preds = []
    with torch.inference_mode():
        for batch in dataloader:
            if hasattr(batch, "bmg"):
                bmg = batch.bmg
                V_d = batch.V_d
                X_d = batch.X_d
            else:
                bmg, V_d, X_d, *_ = batch
            if bmg is None:
                raise RuntimeError("Chemprop batch missing molecular graph (bmg).")
            if hasattr(bmg, "to"):
                moved = bmg.to(device)
                if moved is not None:
                    bmg = moved
            if V_d is not None:
                V_d = V_d.to(device)
            if X_d is not None:
                X_d = X_d.to(device)
            batch_preds = model(bmg, V_d, X_d).detach().cpu().numpy()
            preds.append(batch_preds)

    return np.concatenate(preds, axis=0).squeeze()


def _chemprop_train_v2(
    dataset_type: str,
    train_smiles: list[str],
    val_smiles: list[str],
    fingerprint_type: str | None,
    train_fingerprints: np.ndarray | None,
    val_fingerprints: np.ndarray | None,
    property_name: str,
    train_properties: list[int],
    val_properties: list[int],
    epochs: int,
    save_path: Path,
    num_workers: int = 0,
    use_gpu: bool = False,
) -> Any:
    from chemprop import data, featurizers, nn, models
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from chemprop.models.utils import save_model

    seed_everything(0, workers=True)

    x_d_train = None
    x_d_val = None
    if fingerprint_type is not None:
        x_d_train = [fp.astype(np.float32) for fp in train_fingerprints]
        x_d_val = [fp.astype(np.float32) for fp in val_fingerprints]

    train_datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y=[float(y)], x_d=x_d)
        for smi, y, x_d in zip(train_smiles, train_properties, x_d_train or [None] * len(train_smiles))
    ]
    val_datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y=[float(y)], x_d=x_d)
        for smi, y, x_d in zip(val_smiles, val_properties, x_d_val or [None] * len(val_smiles))
    ]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dataset = data.MoleculeDataset(train_datapoints, featurizer=featurizer)
    val_dataset = data.MoleculeDataset(val_datapoints, featurizer=featurizer)

    train_loader = data.build_dataloader(
        train_dataset, batch_size=min(50, len(train_smiles)), shuffle=True, num_workers=num_workers
    )
    val_loader = data.build_dataloader(
        val_dataset, batch_size=min(50, len(val_smiles)), shuffle=False, num_workers=num_workers
    )

    message_passing = nn.BondMessagePassing()
    aggregation = nn.MeanAggregation()
    input_dim = None
    if fingerprint_type is not None:
        input_dim = 300 + train_fingerprints.shape[1]

    if dataset_type == "classification":
        predictor = nn.BinaryClassificationFFN(n_tasks=1, input_dim=input_dim) if input_dim is not None else nn.BinaryClassificationFFN(n_tasks=1)
    elif dataset_type == "regression":
        predictor = nn.RegressionFFN(n_tasks=1, input_dim=input_dim) if input_dim is not None else nn.RegressionFFN(n_tasks=1)
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported.")

    model = models.MPNN(message_passing, aggregation, predictor)

    checkpoint = ModelCheckpoint(
        dirpath=str(save_path.parent),
        filename=save_path.stem,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        logger=False,
        enable_model_summary=False,
        callbacks=[checkpoint],
        log_every_n_steps=50,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint.best_model_path
    if best_path:
        model = model.load_from_checkpoint(best_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(path=save_path, model=model)

    return model


# ----------------------------
# Public API (versioned)
# ----------------------------

def chemprop_predict(
    model: Any,
    smiles: list[str],
    fingerprints: np.ndarray | None = None,
    num_workers: int = 0,
    chemprop_version: str | None = None,
) -> np.ndarray:
    """Predicts molecular properties using a Chemprop model."""
    version = _normalize_chemprop_version(chemprop_version)
    if version == "v1":
        return _chemprop_predict_v1(
            model=model,
            smiles=smiles,
            fingerprints=fingerprints,
            num_workers=num_workers,
        )

    return _chemprop_predict_v2(
        model=model,
        smiles=smiles,
        fingerprints=fingerprints,
        num_workers=num_workers,
    )


def chemprop_train(
    dataset_type: str,
    train_smiles: list[str],
    val_smiles: list[str],
    fingerprint_type: str | None,
    train_fingerprints: np.ndarray | None,
    val_fingerprints: np.ndarray | None,
    property_name: str,
    train_properties: list[int],
    val_properties: list[int],
    epochs: int,
    save_path: Path,
    num_workers: int = 0,
    use_gpu: bool = False,
    chemprop_version: str | None = None,
) -> Any:
    """Trains and saves a Chemprop model."""
    version = _normalize_chemprop_version(chemprop_version)
    if version == "v1":
        return _chemprop_train_v1(
            dataset_type=dataset_type,
            train_smiles=train_smiles,
            val_smiles=val_smiles,
            fingerprint_type=fingerprint_type,
            train_fingerprints=train_fingerprints,
            val_fingerprints=val_fingerprints,
            property_name=property_name,
            train_properties=train_properties,
            val_properties=val_properties,
            epochs=epochs,
            save_path=save_path,
            num_workers=num_workers,
            use_gpu=use_gpu,
        )

    return _chemprop_train_v2(
        dataset_type=dataset_type,
        train_smiles=train_smiles,
        val_smiles=val_smiles,
        fingerprint_type=fingerprint_type,
        train_fingerprints=train_fingerprints,
        val_fingerprints=val_fingerprints,
        property_name=property_name,
        train_properties=train_properties,
        val_properties=val_properties,
        epochs=epochs,
        save_path=save_path,
        num_workers=num_workers,
        use_gpu=use_gpu,
    )
