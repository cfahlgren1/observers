from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from observers import wrap_docling


# Configure PDF pipeline options
pipeline_options = PdfPipelineOptions(
    images_scale=2.0,
    generate_page_images=True,
    generate_picture_images=True,
    generate_table_images=True,
)

# Set format options for PDF input
format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}

# Initialize and wrap document converter
converter = DocumentConverter(format_options=format_options)
converter = wrap_docling(converter, media_types=["pictures", "tables"])

# Convert single and multiple documents
url = "https://arxiv.org/pdf/2408.09869"
converted = converter.convert(url)
converted = converter.convert_all([url])
