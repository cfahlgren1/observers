import atexit
import json
import os
import uuid
import tempfile
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from huggingface_hub import CommitScheduler, login, metadata_update, whoami

from observers.stores.base import Store


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
        try:
            whoami(token=self.token or os.getenv("HF_TOKEN"))
        except Exception:
            login()

        self._temp_dir = tempfile.mkdtemp(prefix="observers_dataset_")

        if self.folder_path is None:
            self.folder_path = self._temp_dir

        atexit.register(self._cleanup)

    def _cleanup(self):
        """Clean up temporary directory on exit"""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir)

    def _init_table(self, record: "Record"):
        repo_name = self.repo_name or record.table_name
        org_name = self.org_name or whoami(token=self.token)["name"]
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
        metadata_update(
            repo_id=repo_id,
            metadata={"tags": ["observers"]},
            repo_type="dataset",
        )

        atexit.register(self._scheduler.push_to_hub)

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
                record_dict["synced_at"] = None

                for json_field in record.json_fields:
                    if record_dict[json_field]:
                        record_dict[json_field] = json.dumps(record_dict[json_field])

                f.write(json.dumps(record_dict))
