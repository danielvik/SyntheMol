"""Add QED and cLogP columns to a building-block CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from rdkit.Chem.QED import qed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--smiles_column", type=str, default="smiles")
    parser.add_argument("--qed_column", type=str, default="qed")
    parser.add_argument("--clogp_column", type=str, default="clogp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    RDLogger.DisableLog("rdApp.*")

    df = pd.read_csv(args.input_csv)
    if args.smiles_column not in df.columns:
        raise ValueError(
            f"Missing SMILES column '{args.smiles_column}' in {args.input_csv}."
        )

    qed_values: list[float | None] = []
    clogp_values: list[float | None] = []
    invalid = 0

    for smiles in df[args.smiles_column].astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid += 1
            qed_values.append(None)
            clogp_values.append(None)
            continue

        qed_values.append(float(qed(mol)))
        clogp_values.append(float(Crippen.MolLogP(mol)))

    df[args.qed_column] = qed_values
    df[args.clogp_column] = clogp_values
    df.to_csv(args.output_csv, index=False)

    print(f"Wrote {args.output_csv}")
    print(f"Rows: {len(df)}")
    print(f"Invalid SMILES: {invalid}")


if __name__ == "__main__":
    main()
