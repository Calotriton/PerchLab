"""Persist Perch embeddings to a Perch Hoplite SQLite DB (with optional export).

Reuses Hoplite's ``SQLiteUSearchDB`` (the same store the agile-modeling tools
use) so the embeddings are immediately usable for vector search and downstream
classifier training. When the input is organised as one folder per species, the
folder name is attached as a positive annotation (the training-ready path).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audio import RecordingMeta
from .inference import WindowResult
from .logging import get_logger

_log = get_logger("embedding")


@dataclass
class EmbeddingRunner:
    """Write per-window embeddings and labels into a Hoplite DB."""

    db_dir: Path
    embedding_dim: int
    labeled: bool = False
    _db: Any = field(default=None, init=False, repr=False)
    _export_rows: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Open (creating if needed) the Hoplite SQLite/USearch database."""
        from perch_hoplite.db import sqlite_usearch_impl  # noqa: PLC0415

        self.db_dir = Path(self.db_dir)
        usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(self.embedding_dim)
        self._db = sqlite_usearch_impl.SQLiteUSearchDB.create(
            db_path=str(self.db_dir), usearch_cfg=usearch_cfg
        )

    def add_file(
        self,
        meta: RecordingMeta,
        windows: Iterable[WindowResult],
        *,
        collect_export: bool = False,
    ) -> int:
        """Insert one recording's windows (and label) into the DB.

        Args:
            meta: Recording metadata (path, datetime, folder label).
            windows: Per-window inference results to store.
            collect_export: When ``True``, also buffer rows for Parquet/NPZ export.

        Returns:
            Number of windows inserted.
        """
        from perch_hoplite.db import datatypes  # noqa: PLC0415

        recording_id = self._db.insert_recording(
            filename=str(meta.path), datetime=meta.start_datetime
        )
        count = 0
        for window in windows:
            embedding = np.asarray(window.embedding, dtype=np.float32)
            self._db.insert_window(
                recording_id=recording_id,
                offsets=[window.start_s],
                embedding=embedding,
                handle_duplicates="skip",
            )
            if self.labeled and meta.expected_label:
                self._db.insert_annotation(
                    recording_id=recording_id,
                    offsets=[window.start_s],
                    label=meta.expected_label,
                    label_type=datatypes.LabelType.POSITIVE,
                    provenance="perchlab",
                    handle_duplicates="skip",
                )
            if collect_export:
                self._export_rows.append(
                    {
                        "file": str(meta.path),
                        "offset_s": window.start_s,
                        "label": meta.expected_label or "",
                        "device": meta.device or "",
                        "date": meta.date_str,
                        "time": meta.time_str,
                        **{f"e{i}": float(v) for i, v in enumerate(embedding)},
                    }
                )
            count += 1
        return count

    def commit(self) -> None:
        """Flush pending writes to disk."""
        self._db.commit()

    def export(self, path: Path, fmt: str) -> Path:
        """Write buffered embeddings to a portable file.

        Args:
            path: Destination path.
            fmt: ``"parquet"`` or ``"npz"``.

        Returns:
            The written path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._export_rows)
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        elif fmt == "npz":
            emb_cols = [c for c in df.columns if c.startswith("e")]
            np.savez_compressed(
                path,
                embeddings=df[emb_cols].to_numpy(dtype=np.float32),
                labels=df["label"].to_numpy(),
                files=df["file"].to_numpy(),
                offsets=df["offset_s"].to_numpy(),
            )
        else:  # pragma: no cover - guarded by config Literal
            raise ValueError(f"Unknown export format: {fmt}")
        _log.info("Exported %d embeddings to %s", len(df), path)
        return path
