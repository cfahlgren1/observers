import base64
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from PIL import Image

from observers.observers.base import Record
from observers.stores.duckdb import DuckDBStore

if TYPE_CHECKING:
    from argilla import Argilla
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import (
        DoclingDocument,
        ListItem,
        PageItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
    )

    from observers.stores.argilla import ArgillaStore
    from observers.stores.datasets import DatasetsStore


@dataclass
class DoclingRecord(Record):
    """
    Data class for storing Docling API error information
    """

    version: str = None
    mime_type: str = None
    label: str = None
    filename: str = None
    page_no: int = 0
    image: Optional[Image.Image] = None
    mimetype: Optional[str] = None
    dpi: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    uri: Optional[str] = None
    text: Optional[str] = None
    text_length: Optional[int] = None
    raw_response: Dict[str, Any] = None

    @classmethod
    def create(
        cls,
        document: "DoclingDocument",
        docling_object: Union[
            "PictureItem", "TableItem", "ListItem", "TextItem", "SectionHeaderItem"
        ],
        page: Optional[Union["PageItem", int]] = None,
        tags: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> "DoclingRecord":
        data = {}
        # get base info
        data["version"] = document.version
        data["mime_type"] = document.origin.mimetype
        data["filename"] = document.origin.filename
        data["page_no"] = page.page_no if not isinstance(page, int) else page
        data["label"] = docling_object.label.value
        # get image info
        if hasattr(docling_object, "image"):
            data["image"] = docling_object.image.pil_image
        else:
            data["image"] = docling_object.get_image(document)
        if data["image"]:
            data["mimetype"] = "image/png"  # PIL images are saved as PNG
            data["dpi"] = data["image"].info.get(
                "dpi", 72
            )  # Default to 72 DPI if not specified
            data["width"] = data["image"].width
            data["height"] = data["image"].height
            # Create data URI for the image
            buffered = BytesIO()
            data["image"].save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data["uri"] = f"data:image/png;base64,{img_str}"

        # get caption or text
        if hasattr(docling_object, "caption_text") and callable(
            docling_object.caption_text
        ):
            data["text"] = docling_object.caption_text(document)
        if getattr(docling_object, "export_to_document_tokens") and callable(
            docling_object.export_to_document_tokens
        ):
            data["text"] = docling_object.export_to_document_tokens(document)
        data["text_length"] = len(data["text"])
        data["raw_response"] = docling_object.model_dump(mode="json")
        return cls(**data, tags=tags, properties=properties, error=error)

    @property
    def table_name(self):
        return "docling_records"

    @property
    def json_fields(self):
        return ["raw_response", "properties"]

    @property
    def image_fields(self):
        return ["image"]

    @property
    def table_columns(self):
        return [
            "id",
            "version",
            "mime_type",
            "page_no",
            "image",
            "filename",
            "label",
            "mimetype",
            "dpi",
            "width",
            "height",
            "uri",
            "text",
            "text_length",
            "tags",
            "properties",
            "error",
            "raw_response",
            "synced_at",
        ]

    @property
    def duckdb_schema(self):
        return f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR PRIMARY KEY,
            version VARCHAR,
            mime_type VARCHAR,
            page_no INTEGER,
            image BLOB,
            filename VARCHAR,
            label VARCHAR,
            mimetype VARCHAR,
            dpi INTEGER,
            width INTEGER,
            height INTEGER,
            uri VARCHAR,
            text VARCHAR,
            text_length INTEGER,
            tags VARCHAR[],
            properties JSON,
            error VARCHAR,
            raw_response JSON,
            synced_at TIMESTAMP
        )
        """

    def argilla_settings(self, client: "Argilla"):
        import argilla as rg
        from argilla import Settings

        return Settings(
            fields=[
                rg.ImageField(
                    name="uri",
                    description="The image.",
                    _client=client,
                    required=False,
                ),
                rg.TextField(
                    name="text",
                    description="The caption text.",
                    markdown=True,
                    required=False,
                    client=client,
                ),
            ],
            questions=[
                rg.TextQuestion(
                    name="question_or_query",
                    title="Question or Query",
                    description="The question or query associated with the picture.",
                    required=True,
                ),
                rg.TextQuestion(
                    name="answer",
                    title="Answer",
                    description="The answer to the question or query associated with the picture.",
                    required=False,
                ),
                rg.RatingQuestion(
                    name="rating_image",
                    title="Rating image",
                    description="How would you rate the picture? 1 being the least relevant and 5 being the most relevant.",
                    values=[1, 2, 3, 4, 5],
                    required=False,
                ),
                rg.RatingQuestion(
                    name="rating_text",
                    title="Rating text",
                    description="How would you rate the text? 1 being the worst and 5 being the best.",
                    values=[1, 2, 3, 4, 5],
                    required=False,
                ),
                rg.TextQuestion(
                    name="text_improve",
                    title="Improve text",
                    description="If you would like to improve the text, please provide a better text here.",
                    required=False,
                ),
            ],
            metadata=[
                rg.TermsMetadataProperty(name="version", client=client),
                rg.TermsMetadataProperty(name="mime_type", client=client),
                rg.TermsMetadataProperty(name="page_no", client=client),
                rg.TermsMetadataProperty(name="filename", client=client),
                rg.TermsMetadataProperty(name="label", client=client),
                rg.TermsMetadataProperty(name="mimetype", client=client),
                rg.IntegerMetadataProperty(name="dpi", client=client),
                rg.IntegerMetadataProperty(name="width", client=client),
                rg.IntegerMetadataProperty(name="height", client=client),
                rg.TermsMetadataProperty(name="text_length", client=client),
            ],
        )


def wrap_docling(
    client: "DocumentConverter",
    store: Optional[Union["DatasetsStore", "ArgillaStore", DuckDBStore]] = None,
    tags: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
    media_types: Optional[List[str]] = None,
) -> "DocumentConverter":
    """
    Wrap DocumentConverter client to track API calls in a Store.

    Args:
        client: OpenAI client instance
        store: Store instance for persistence. Creates new if None
        tags: Optional list of tags to associate with records
        properties: Optional dictionary of properties to associate with records
        media_type: Optional media type to associate with records "texts", "pictures", "tables" or None for all

    Returns:
        DocumentConverter: Wrapped DocumentConverter client
    """
    from docling_core.types.doc.document import (
        ListItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
    )

    if store is None:
        store = DuckDBStore.connect()
    tags = tags or []
    properties = properties or {}

    if media_types is None:
        media_types = ["texts", "pictures", "tables"]
    elif any(
        media_type not in ["texts", "pictures", "tables"] for media_type in media_types
    ):
        raise ValueError(f"Invalid media type: {media_types}")

    original_convert = client.convert

    def convert(*args, **kwargs) -> "DoclingDocument":
        result = original_convert(*args, **kwargs)
        document = result.document
        for page_no, page in enumerate(document.pages):
            for docling_object, _level in document.iterate_items(page_no=page_no):
                if (
                    isinstance(docling_object, (SectionHeaderItem, ListItem, TextItem))
                    and "texts" in media_types
                ):
                    record = DoclingRecord.create(
                        docling_object=docling_object,
                        document=document,
                        page=page,
                        tags=tags,
                        properties=properties,
                    )
                    store.add(record)
                if (
                    isinstance(docling_object, PictureItem)
                    and "pictures" in media_types
                ):
                    record = DoclingRecord.create(
                        docling_object=docling_object,
                        document=document,
                        page=page,
                        tags=tags,
                        properties=properties,
                    )
                    store.add(record)
                if isinstance(docling_object, TableItem) and "tables" in media_types:
                    record = DoclingRecord.create(
                        docling_object=docling_object,
                        document=document,
                        page=page,
                        tags=tags,
                        properties=properties,
                    )
                    store.add(record)

        return record

    client.convert = convert
    return client