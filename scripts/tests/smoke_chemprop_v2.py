"""Smoke test for Chemprop v2 integration.

This trains a tiny model on a toy dataset and runs a prediction pass.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from chemfunc import compute_fingerprints

from scripts.models.chemprop_models import chemprop_predict, chemprop_train


def main() -> None:
    # Tiny dataset
    data = pd.DataFrame(
        {
            "smiles": [
                "CCO",
                "CCN",
                "CCC",
                "c1ccccc1",
                "CC(=O)O",
                "COC",
                "CCCl",
                "CCBr",
            ],
            "activity": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_path = tmpdir / "model_0.pt"

        fingerprints = compute_fingerprints(data["smiles"], fingerprint_type="rdkit")

        # Simple train/val split
        train_idx = np.arange(len(data))[:6]
        val_idx = np.arange(len(data))[6:]

        train_smiles = data.loc[train_idx, "smiles"].tolist()
        val_smiles = data.loc[val_idx, "smiles"].tolist()

        train_props = data.loc[train_idx, "activity"].tolist()
        val_props = data.loc[val_idx, "activity"].tolist()

        train_fps = fingerprints[train_idx]
        val_fps = fingerprints[val_idx]

        model = chemprop_train(
            dataset_type="classification",
            train_smiles=train_smiles,
            val_smiles=val_smiles,
            fingerprint_type="rdkit",
            train_fingerprints=train_fps,
            val_fingerprints=val_fps,
            property_name="activity",
            train_properties=train_props,
            val_properties=val_props,
            epochs=1,
            save_path=model_path,
            num_workers=0,
            use_gpu=False,
            chemprop_version="v2",
        )

        preds = chemprop_predict(
            model=model,
            smiles=data["smiles"].tolist(),
            fingerprints=fingerprints,
            num_workers=0,
            chemprop_version="v2",
        )

        print("Preds shape:", preds.shape)
        print("Preds min/max:", float(np.min(preds)), float(np.max(preds)))
        print("OK")


if __name__ == "__main__":
    main()
