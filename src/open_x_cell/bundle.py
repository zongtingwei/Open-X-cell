"""Portable inference weights and the exact fixed priors used with them."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from safetensors.torch import save_file

from open_x_cell.priors import PRIOR_SOURCES, _load_prior_table


BUNDLE_FILES = {"config.json", "genes.json", "model.safetensors"} | {
    f"priors/{source}.npz" for source in PRIOR_SOURCES
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(directory: str | Path) -> dict:
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "open-x-cell-v1":
        raise ValueError("unsupported Open-X-Cell bundle format")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != BUNDLE_FILES:
        raise ValueError("bundle files must include weights, configuration, vocabulary and all five priors")
    for name, record in files.items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f"bundle file escapes its directory: {name}")
        if not path.is_file():
            raise ValueError(f"missing bundle file: {name}")
        if path.stat().st_size != record.get("bytes"):
            raise ValueError(f"bundle file size mismatch: {name}")
        if sha256(path) != record.get("sha256"):
            raise ValueError(f"bundle checksum mismatch: {name}")
    return manifest


def export_bundle(
    checkpoint: str | Path,
    output: str | Path,
    prior_files: dict[str, str | Path],
    *,
    trust_legacy_pickle: bool = False,
    metadata: dict | None = None,
) -> Path:
    if not trust_legacy_pickle:
        raise ValueError("export requires trust_legacy_pickle=True for your trusted training checkpoint")
    if set(prior_files) != set(PRIOR_SOURCES) or any(v is None for v in prior_files.values()):
        raise ValueError("supply the exact five prior files used with this checkpoint")
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(target)
    from open_x_cell.training import load_checkpoint
    model, config, genes = load_checkpoint(checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xcell-export-", dir=target.parent) as tmp:
        stage = Path(tmp) / "bundle"
        (stage / "priors").mkdir(parents=True)
        (stage / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
        (stage / "genes.json").write_text(json.dumps(genes, indent=2) + "\n")
        save_file({key: value.detach().cpu().contiguous().clone()
                   for key, value in model.state_dict().items()}, str(stage / "model.safetensors"))
        counts = {}
        for source in PRIOR_SOURCES:
            table = _load_prior_table(prior_files[source], config.prior_dims[source])
            kept = [gene for gene in genes if gene in table]
            if not kept:
                raise ValueError(f"prior {source} has no genes in the checkpoint vocabulary")
            values = np.stack([table[gene] for gene in kept])
            np.savez_compressed(stage / "priors" / f"{source}.npz",
                                genes=np.asarray(kept), embeddings=values)
            counts[source] = len(kept)
            del table, values
        manifest = {
            "format": "open-x-cell-v1",
            "parameters": sum(p.numel() for p in model.parameters()),
            "source_checkpoint_sha256": sha256(checkpoint),
            "source_prior_sha256": {key: sha256(value) for key, value in prior_files.items()},
            "prior_gene_counts": counts,
            "metadata": metadata or {},
            "files": {name: {"bytes": (stage / name).stat().st_size,
                             "sha256": sha256(stage / name)} for name in sorted(BUNDLE_FILES)},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_bundle(stage)
        stage.rename(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export trusted Open-X-Cell weights and fixed priors.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-legacy-pickle", action="store_true")
    for source in PRIOR_SOURCES:
        parser.add_argument("--" + source.replace("_", "-") + "-priors", required=True)
    args = parser.parse_args(argv)
    path = export_bundle(args.checkpoint, args.output,
                         {source: getattr(args, source + "_priors") for source in PRIOR_SOURCES},
                         trust_legacy_pickle=args.trust_legacy_pickle)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
