import base64
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

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
    image: Optional[Dict[str, Any]] = None
    mimetype: Optional[str] = None
    dpi: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    uri: Optional[str] = None
    text: Optional[str] = None
    caption_text: Optional[str] = None
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
        image = None
        try:
            if hasattr(docling_object, "image"):
                image = docling_object.image.pil_image
                docling_object.image.uri = None
            else:
                image = docling_object.get_image(document)
            if image:
                data["mimetype"] = "image/png"  # PIL images are saved as PNG
                data["dpi"] = image.info.get(
                    "dpi", 72
                )  # Default to 72 DPI if not specified
                data["width"] = image.width
                data["height"] = image.height
                # Create data URI for the image
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data["image"] = {"bytes": img_str, "path": None}
        except Exception as e:
            error = str(e)

        # get caption from image or table
        caption_text = None
        try:
            if hasattr(docling_object, "caption_text") and callable(
                docling_object.caption_text
            ):
                caption_text = docling_object.caption_text(document)
                data["caption_text"] = caption_text if caption_text else None
        except Exception as e:
            error = str(e)

        # get table as html
        if hasattr(docling_object, "export_to_dataframe") and callable(
            docling_object.export_to_dataframe
        ):
            try:
                data["text"] = docling_object.export_to_html(
                    document, add_caption=False
                )
            except Exception as e:
                error = str(e)

        # get text from item
        if hasattr(docling_object, "text"):
            data["text"] = docling_object.text

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
    def text_fields(self):
        return ["text", "caption_text"]

    @property
    def table_columns(self):
        return [
            "id",
            "filename",
            "label",
            "text",
            "caption_text",
            "image",
            "width",
            "height",
            "dpi",
            "mimetype",
            "page_no",
            "mime_type",
            "version",
            "tags",
            "properties",
            "error",
            "raw_response",
        ]

    @property
    def duckdb_schema(self):
        return f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR PRIMARY KEY,
            filename VARCHAR,
            label VARCHAR,
            text VARCHAR,
            caption_text VARCHAR,
            image STRUCT(bytes BLOB, path VARCHAR),
            width INTEGER,
            height INTEGER,
            dpi INTEGER,
            mimetype VARCHAR,
            page_no INTEGER,
            mime_type VARCHAR,
            version VARCHAR,
            tags VARCHAR[],
            properties JSON,
            error VARCHAR,
            raw_response JSON,
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
                    description="The text.",
                    markdown=True,
                    required=False,
                    client=client,
                ),
                rg.TextField(
                    name="caption_text",
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
                    description="The question or query associated with the information.",
                    required=True,
                ),
                rg.TextQuestion(
                    name="answer",
                    title="Answer",
                    description="The answer to the question or query associated with the information.",
                    required=False,
                ),
                rg.RatingQuestion(
                    name="rating_image",
                    title="Rating image",
                    description="How would you rate the image? 1 being the least relevant and 5 being the most relevant.",
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
                rg.TextQuestion(
                    name="caption_text_improve",
                    title="Improve caption text",
                    description="If you would like to improve the caption text, please provide a better caption text here.",
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
                rg.IntegerMetadataProperty(name="text_length", client=client),
                rg.IntegerMetadataProperty(name="caption_text_length", client=client),
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
    original_convert_all = client.convert_all

    def process_document(document, page_no, page) -> None:
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
            if isinstance(docling_object, PictureItem) and "pictures" in media_types:
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

    def convert(*args, **kwargs) -> "DoclingDocument":
        result = original_convert(*args, **kwargs)
        document = result.document
        for page_no, page in enumerate(document.pages):
            process_document(document, page_no, page)
        return result

    def convert_all(*args, **kwargs) -> Iterator["DoclingDocument"]:
        results = original_convert_all(*args, **kwargs)
        for result in results:
            document = result.document
            for page_no, page in enumerate(document.pages):
                process_document(document, page_no, page)
            yield result

    client.convert = convert
    client.convert_all = convert_all
    return client
