import json
import logging
import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

try:
    import fitz
except (ImportError, OSError) as exc:
    fitz = None
    logger.warning("PyMuPDF renderer is unavailable; using pypdf for embedded text and images: %s", exc)


class LocalPdf:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stream = None
        self._document = None
        if fitz is not None:
            try:
                self._document = fitz.open(self.path)
            except (fitz.FileDataError, fitz.EmptyFileError, OSError, ValueError) as exc:
                raise ValueError("PDF could not be opened.") from exc
            self.page_count = len(self._document)
        else:
            try:
                self._stream = self.path.open("rb")
                self._document = PdfReader(self._stream, strict=False)
                if self._document.is_encrypted:
                    self.close()
                    raise ValueError("Password-protected PDFs are not supported.")
                self.page_count = len(self._document.pages)
            except (PdfReadError, OSError) as exc:
                self.close()
                raise ValueError("PDF could not be opened.") from exc

    @property
    def engine(self) -> str:
        return "pymupdf" if fitz is not None else "pypdf"

    def text(self, index: int) -> str:
        if fitz is not None:
            return self._document.load_page(index).get_text("text", sort=True)
        return self._document.pages[index].extract_text() or ""

    def embedded_images(self, index: int) -> list[tuple[bytes, str]]:
        if fitz is not None:
            return []
        images = self._document.pages[index].images
        found = []
        for image in images:
            data = image.data
            suffix = Path(image.name).suffix.casefold() or ".bin"
            found.append((data, suffix))
        return found

    def render_png(self, index: int, dpi: int = 140) -> bytes | None:
        if fitz is None:
            return None
        return self._document.load_page(index).get_pixmap(dpi=dpi, alpha=False).tobytes("png")

    def rotation(self, index: int) -> int:
        if fitz is not None:
            return self._document.load_page(index).rotation % 360
        return int(self._document.pages[index].get("/Rotate", 0) or 0) % 360

    def close(self) -> None:
        if self._document is not None and fitz is not None:
            self._document.close()
        if self._stream is not None:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class LocalImageBundle:
    """Read privately stored, validated image pages through the PDF pipeline interface."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        try:
            filenames = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Image bundle manifest could not be opened.") from exc
        if not isinstance(filenames, list) or not filenames:
            raise ValueError("Image bundle contains no pages.")
        if any(not isinstance(name, str) or not re.fullmatch(r"page-\d+-image-0\.(?:png|jpe?g)", name) for name in filenames):
            raise ValueError("Image bundle manifest contains an invalid page path.")
        self.paths = [self.manifest_path.parent / "page_images" / name for name in filenames]
        image_dir = (self.manifest_path.parent / "page_images").resolve()
        if any(not path.is_file() or path.is_symlink() or path.resolve().parent != image_dir for path in self.paths):
            raise ValueError("An uploaded image page is unavailable.")
        self.page_count = len(self.paths)

    @property
    def engine(self) -> str:
        return "image_upload"

    def text(self, index: int) -> str:
        return ""

    def embedded_images(self, index: int) -> list[tuple[bytes, str]]:
        path = self.paths[index]
        return [(path.read_bytes(), path.suffix.casefold())]

    def render_png(self, index: int, dpi: int = 140) -> bytes | None:
        return None

    def rotation(self, index: int) -> int:
        return 0

    def source_image_path(self, index: int) -> Path:
        return self.paths[index]

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
