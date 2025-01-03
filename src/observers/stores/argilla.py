import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

import argilla as rg
from argilla import (
    Argilla,
    LabelQuestion,
    MultiLabelQuestion,
    RankingQuestion,
    RatingQuestion,
    SpanQuestion,
    TextQuestion,
)

from observers.stores.base import Store


if TYPE_CHECKING:
    from observers.base import Record


@dataclass
class ArgillaStore(Store):
    """
    Argilla store
    """

    api_url: Optional[str] = field(default=None)
    api_key: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default=None)
    workspace_name: Optional[str] = field(default=None)
    questions: Optional[
        List[
            Union[
                TextQuestion,
                LabelQuestion,
                SpanQuestion,
                RatingQuestion,
                RankingQuestion,
                MultiLabelQuestion,
            ]
        ]
    ] = field(default=None)

    _dataset: Optional[rg.Dataset] = None
    _dataset_keys: Optional[List[str]] = None
    _client: Optional[Argilla] = None

    def __post_init__(self) -> None:
        """Initialize the store"""
        self._client = Argilla(api_url=self.api_url, api_key=self.api_key)

    def _init_table(self, record: "Record") -> None:
        dataset_name = (
            self.dataset_name or f"{record.table_name}_{uuid.uuid4().hex[:8]}"
        )
        workspace_name = self.workspace_name or self._client.me.username
        workspace = self._client.workspaces(name=workspace_name)
        if not workspace:
            workspace = self._client.workspaces.add(rg.Workspace(name=workspace_name))
        dataset = self._client.datasets(name=dataset_name, workspace=workspace_name)

        if not dataset:
            settings = record.argilla_settings(self._client)
            if self.questions:
                settings.questions = self.questions
            dataset = rg.Dataset(
                name=dataset_name,
                workspace=workspace_name,
                settings=settings,
                client=self._client,
            ).create()
        elif self.questions:
            raise ValueError(
                "Custom questions are not supported for existing datasets."
            )
        self._dataset = dataset
        dataset_keys = (
            [field.name for field in dataset.settings.fields]
            + [question.name for question in dataset.settings.questions]
            + [term.name for term in dataset.settings.metadata]
            + [vector.name for vector in dataset.settings.vectors]
        )
        self._dataset_keys = dataset_keys

    @classmethod
    def connect(
        cls,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dataset_name: Optional[str] = None,
        workspace_name: Optional[str] = None,
    ) -> "ArgillaStore":
        """Create a new store instance with custom settings"""
        return cls(
            api_url=api_url,
            api_key=api_key,
            dataset_name=dataset_name,
            workspace_name=workspace_name,
        )

    def add(self, record: "Record") -> None:
        """Add a new record to the database"""
        if not self._dataset:
            self._init_table(record)

        record_dict = asdict(record)

        for text_field in record.text_fields:
            if text_field in record_dict:
                record_dict[f"{text_field}_length"] = len(record_dict[text_field])

        record_dict = {k: v for k, v in record_dict.items() if k in self._dataset_keys}
        self._dataset.records.log([record_dict])

    async def add_async(self, record: "Record"):
        """Add a new record to the database asynchronously"""
        if not self._dataset:
            self._init_table(record)

        record_dict = asdict(record)

        for text_field in record.text_fields:
            if text_field in record_dict:
                record_dict[f"{text_field}_length"] = len(record_dict[text_field])

        record_dict = {k: v for k, v in record_dict.items() if k in self._dataset_keys}
        # Use argilla's native async API
        await self._dataset.records.alog(
            [record_dict],
            background=True,
            verbose=False,
        )
