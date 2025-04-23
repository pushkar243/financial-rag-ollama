from PyPDF2 import PdfReader
import logging

logger = logging.getLogger(__name__)

class PDFLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> str:
        logger.info(f"Loading PDF: {self.filepath}")
        with open(self.filepath, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text