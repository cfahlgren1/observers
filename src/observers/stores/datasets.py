import atexit
import base64
import hashlib
import json
import os
import tempfile
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from datasets import Dataset
from datasets import Image as DatasetImage
from datasets.utils.logging import disable_progress_bar
from huggingface_hub import CommitScheduler, login, metadata_update, upload_file, whoami
from PIL import Image

from observers.stores.base import Store

if TYPE_CHECKING:
    from observers.observers.base import Record


disable_progress_bar()


def push_to_hub(self):
    """Push pending changes to the Hugging Face Hub"""
    json_files = Path(self.folder_path).rglob("*.json")
    records = [json.loads(line) for json_file in json_files for line in open(json_file)]

    if records:
        image_keys: List[json.Any] = [
            key
            for key in records[0].keys()
            if isinstance(records[0][key], dict) and "path" in records[0][key]
        ]

        for record in records:
            for key in image_keys:
                record[key] = str(Path(self.folder_path) / record[key]["path"])

        dataset = Dataset.from_list(records)
        for key in image_keys:
            dataset = dataset.cast_column(key, DatasetImage())

        with self.lock:
            buffer = BytesIO()
            dataset.to_parquet(buffer)
            buffer.seek(0)
            random_id = uuid.uuid4().hex
            filename = f"data/train-{random_id}.parquet"
            upload_file(
                path_or_fileobj=buffer,
                path_in_repo=filename,
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=f"Upload {filename}",
            )

            # Delete all json files
            for json_file in json_files:
                try:
                    json_file.unlink()
                except Exception as e:
                    warnings.warn(f"Failed to delete {json_file}: {e}")


CommitScheduler.push_to_hub = push_to_hub


@dataclass
class DatasetsStore(Store):
    """
    Datasets store
    """

    org_name: Optional[str] = field(default=None)
    repo_name: Optional[str] = field(default=None)
    folder_path: Optional[str] = field(default=None)
    every: Optional[int] = field(default=5)
    path_in_repo: Optional[str] = field(default=None)
    revision: Optional[str] = field(default=None)
    private: Optional[bool] = field(default=None)
    token: Optional[str] = field(default=None)
    allow_patterns: Optional[List[str]] = field(default=None)
    ignore_patterns: Optional[List[str]] = field(default=None)
    squash_history: Optional[bool] = field(default=None)

    _filename: Optional[str] = field(default=None)
    _scheduler: Optional[CommitScheduler] = None
    _temp_dir: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize the store and create temporary directory"""
        if self.ignore_patterns is None:
            self.ignore_patterns = ["*.json"]

        try:
            whoami(token=self.token or os.getenv("HF_TOKEN"))
        except Exception:
            login()

        if self.folder_path is None:
            self._temp_dir = tempfile.mkdtemp(prefix="observers_dataset_")
            self.folder_path = self._temp_dir
            atexit.register(self._cleanup)
        else:
            os.makedirs(self.folder_path, exist_ok=True)

    def _cleanup(self):
        """Clean up temporary directory on exit"""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil

            shutil.rmtree(self._temp_dir)

    def _init_table(self, record: "Record"):
        import logging

        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        repo_name = self.repo_name or f"{record.table_name}_{uuid.uuid4().hex[:8]}"
        org_name = self.org_name or whoami(token=self.token).get("name")
        repo_id = f"{org_name}/{repo_name}"
        self._filename = f"{record.table_name}_{uuid.uuid4()}.json"
        self._scheduler = CommitScheduler(
            repo_id=repo_id,
            folder_path=self.folder_path,
            every=self.every,
            path_in_repo=self.path_in_repo,
            repo_type="dataset",
            revision=self.revision,
            private=self.private,
            token=self.token,
            allow_patterns=self.allow_patterns,
            ignore_patterns=self.ignore_patterns,
            squash_history=self.squash_history,
        )
        self._scheduler.private = self.private
        metadata_update(
            repo_id=repo_id,
            metadata={"tags": ["observers", record.table_name.split("_")[0]]},
            repo_type="dataset",
            token=self.token,
            overwrite=True,
        )

    @classmethod
    def connect(
        cls,
        org_name: Optional[str] = None,
        repo_name: Optional[str] = None,
        folder_path: Optional[str] = None,
        every: Optional[int] = 5,
        path_in_repo: Optional[str] = None,
        revision: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        squash_history: Optional[bool] = None,
    ) -> "DatasetsStore":
        """Create a new store instance with optional custom path"""
        return cls(
            org_name=org_name,
            repo_name=repo_name,
            folder_path=folder_path,
            every=every,
            path_in_repo=path_in_repo,
            revision=revision,
            private=private,
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            squash_history=squash_history,
        )

    def add(self, record: "Record"):
        """Add a new record to the database"""
        if not self._scheduler:
            self._init_table(record)

        with self._scheduler.lock:
            with (self._scheduler.folder_path / self._filename).open("a") as f:
                record_dict = asdict(record)

                # Handle JSON fields
                for json_field in record.json_fields:
                    if record_dict[json_field]:
                        record_dict[json_field] = json.dumps(record_dict[json_field])

                # Handle image fields
                for image_field in record.image_fields:
                    if record_dict[image_field]:
                        image_folder = self._scheduler.folder_path / "images"
                        image_folder.mkdir(exist_ok=True)

                        # Generate unique filename based on record content
                        filtered_dict = {
                            k: v
                            for k, v in sorted(record_dict.items())
                            if k not in ["uri", image_field, "id"]
                        }
                        content_hash = hashlib.sha256(
                            json.dumps(obj=filtered_dict, sort_keys=True).encode()
                        ).hexdigest()
                        image_path = image_folder / f"{content_hash}.png"

                        # Save image and update record
                        image_bytes = base64.b64decode(
                            record_dict[image_field]["bytes"]
                        )
                        Image.open(BytesIO(image_bytes)).save(image_path)
                        record_dict[image_field].update(
                            {"path": str(image_path), "bytes": None}
                        )

                # Clean up empty dictionaries
                record_dict = {
                    k: None if v == {} else v for k, v in record_dict.items()
                }
                sorted_dict = {
                    col: record_dict.get(col) for col in record.table_columns
                }
                try:
                    f.write(json.dumps(sorted_dict) + "\n")
                    f.flush()
                except Exception:
                    raise
