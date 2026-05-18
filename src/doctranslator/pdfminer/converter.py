import io
import logging
import re
from collections.abc import Sequence
from typing import BinaryIO
from typing import Generic
from typing import TextIO
from typing import TypeVar
from typing import cast

from src.doctranslator.pdfminer import utils
from src.doctranslator.pdfminer.image import ImageWriter
from src.doctranslator.pdfminer.layout import LAParams
from src.doctranslator.pdfminer.layout import LTAnno
from src.doctranslator.pdfminer.layout import LTChar
from src.doctranslator.pdfminer.layout import LTComponent
from src.doctranslator.pdfminer.layout import LTContainer
from src.doctranslator.pdfminer.layout import LTCurve
from src.doctranslator.pdfminer.layout import LTFigure
from src.doctranslator.pdfminer.layout import LTImage
from src.doctranslator.pdfminer.layout import LTItem
from src.doctranslator.pdfminer.layout import LTLayoutContainer
from src.doctranslator.pdfminer.layout import LTLine
from src.doctranslator.pdfminer.layout import LTPage
from src.doctranslator.pdfminer.layout import LTRect
from src.doctranslator.pdfminer.layout import LTText
from src.doctranslator.pdfminer.layout import LTTextBox
from src.doctranslator.pdfminer.layout import LTTextBoxVertical
from src.doctranslator.pdfminer.layout import LTTextGroup
from src.doctranslator.pdfminer.layout import LTTextLine
from src.doctranslator.pdfminer.layout import TextGroupElement
from src.doctranslator.pdfminer.pdfcolor import PDFColorSpace
from src.doctranslator.pdfminer.pdfdevice import PDFTextDevice
from src.doctranslator.pdfminer.pdfexceptions import PDFValueError
from src.doctranslator.pdfminer.pdffont import PDFFont
from src.doctranslator.pdfminer.pdffont import PDFUnicodeNotDefined
from src.doctranslator.pdfminer.pdfinterp import PDFGraphicState
from src.doctranslator.pdfminer.pdfinterp import PDFResourceManager
from src.doctranslator.pdfminer.pdfpage import PDFPage
from src.doctranslator.pdfminer.pdftypes import PDFStream
from src.doctranslator.pdfminer.utils import AnyIO
from src.doctranslator.pdfminer.utils import Matrix
from src.doctranslator.pdfminer.utils import PathSegment
from src.doctranslator.pdfminer.utils import Point
from src.doctranslator.pdfminer.utils import Rect
from src.doctranslator.pdfminer.utils import apply_matrix_pt
from src.doctranslator.pdfminer.utils import bbox2str
from src.doctranslator.pdfminer.utils import enc
from src.doctranslator.pdfminer.utils import make_compat_str
from src.doctranslator.pdfminer.utils import mult_matrix

log = logging.getLogger(__name__)


class PDFLayoutAnalyzer(PDFTextDevice):
    cur_item: LTLayoutContainer
    ctm: Matrix

    def __init__(  # NOSONAR - pdfminer-compatible converter API keeps these keyword parameters
        self,
        rsrcmgr: PDFResourceManager,
        pageno: int = 1,
        laparams: LAParams | None = None,
    ) -> None:
        PDFTextDevice.__init__(self, rsrcmgr)
        self.pageno = pageno
        self.laparams = laparams
        self._stack: list[LTLayoutContainer] = []

    def begin_page(self, page: PDFPage, ctm: Matrix) -> None:
        (x0, y0, x1, y1) = page.mediabox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(self.pageno, mediabox)

    def end_page(self, page: PDFPage) -> None:
        assert not self._stack, str(len(self._stack))
        assert isinstance(self.cur_item, LTPage), str(type(self.cur_item))
        if self.laparams is not None:
            self.cur_item.analyze(self.laparams)
        self.pageno += 1
        self.receive_layout(self.cur_item)

    def begin_figure(self, name: str, bbox: Rect, matrix: Matrix) -> None:
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))

    def end_figure(self, _: str) -> None:
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)

    def render_image(self, name: str, stream: PDFStream) -> None:
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        item = LTImage(
            name,
            stream,
            (self.cur_item.x0, self.cur_item.y0, self.cur_item.x1, self.cur_item.y1),
        )
        self.cur_item.add(item)

    def _make_curve(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        pts: list[Point],
        transformed_path: list[PathSegment],
        passthrough_instruction: object,
        xobj_id: object,
        current_clip_paths: object,
        path: Sequence[PathSegment],
    ) -> LTCurve:
        """Build an LTCurve and attach metadata."""
        curve = LTCurve(
            gstate.linewidth,
            pts,
            stroke,
            fill,
            evenodd,
            gstate.scolor,
            gstate.ncolor,
            transformed_path,
            gstate.dash,
        )
        curve.passthrough_instruction = passthrough_instruction
        curve.xobj_id = xobj_id
        curve.render_order = self.il_creater.get_render_order_and_increase()
        curve.ctm = self.ctm
        curve.raw_path = path.copy()
        curve.clip_paths = current_clip_paths
        return curve

    def _paint_line(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        pts: list[Point],
        transformed_path: list[PathSegment],
        passthrough_instruction: object,
        xobj_id: object,
        current_clip_paths: object,
        path: Sequence[PathSegment],
    ) -> None:
        """Paint a single line segment (shape 'mlh' or 'ml')."""
        line = LTLine(
            gstate.linewidth,
            pts[0],
            pts[1],
            stroke,
            fill,
            evenodd,
            gstate.scolor,
            gstate.ncolor,
            original_path=transformed_path,
            dashing_style=gstate.dash,
        )
        line.passthrough_instruction = passthrough_instruction
        line.xobj_id = xobj_id
        line.render_order = self.il_creater.get_render_order_and_increase()
        line.ctm = self.ctm
        line.raw_path = path.copy()
        line.clip_paths = current_clip_paths
        self.cur_item.add(line)

    def _paint_rect_or_curve(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        pts: list[Point],
        transformed_path: list[PathSegment],
        passthrough_instruction: object,
        xobj_id: object,
        current_clip_paths: object,
        path: Sequence[PathSegment],
    ) -> None:
        """Paint a rectangle if pts form a closed rectangle, otherwise a curve."""
        (x0, y0), (x1, y1), (x2, y2), (x3, y3), _ = pts
        is_closed_loop = pts[0] == pts[4]
        has_square_coordinates = (x0 == x1 and y1 == y2 and x2 == x3 and y3 == y0) or (
            y0 == y1 and x1 == x2 and y2 == y3 and x3 == x0
        )
        if is_closed_loop and has_square_coordinates:
            rect = LTRect(
                gstate.linewidth,
                (*pts[0], *pts[2]),
                stroke,
                fill,
                evenodd,
                gstate.scolor,
                gstate.ncolor,
                transformed_path,
                gstate.dash,
            )
            rect.passthrough_instruction = passthrough_instruction
            rect.xobj_id = xobj_id
            rect.render_order = self.il_creater.get_render_order_and_increase()
            rect.ctm = self.ctm
            rect.raw_path = path.copy()
            rect.clip_paths = current_clip_paths
            self.cur_item.add(rect)
        else:
            self.cur_item.add(
                self._make_curve(
                    gstate,
                    stroke,
                    fill,
                    evenodd,
                    pts,
                    transformed_path,
                    passthrough_instruction,
                    xobj_id,
                    current_clip_paths,
                    path,
                )
            )

    def _transform_path(
        self, path: Sequence[PathSegment]
    ) -> tuple[list[Point], list[PathSegment]]:
        """Compute transformed points and path segments for the given path."""
        raw_pts = [cast(Point, p[-2:] if p[0] != "h" else path[0][-2:]) for p in path]
        pts = [apply_matrix_pt(self.ctm, pt) for pt in raw_pts]
        operators = [str(operation[0]) for operation in path]
        transformed_points = [
            [
                apply_matrix_pt(self.ctm, (float(operand1), float(operand2)))
                for operand1, operand2 in zip(
                    operation[1::2], operation[2::2], strict=False
                )
            ]
            for operation in path
        ]
        transformed_path = [
            cast(PathSegment, (o, *p))
            for o, p in zip(operators, transformed_points, strict=False)
        ]
        return pts, transformed_path

    def _dispatch_path_shape(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        path: Sequence[PathSegment],
        shape: str,
        pts: list[Point],
        transformed_path: list[PathSegment],
        passthrough_instruction: object,
        xobj_id: object,
        current_clip_paths: object,
    ) -> None:
        """Dispatch painting to the appropriate helper based on path shape."""
        if shape in {"mlh", "ml"}:
            self._paint_line(
                gstate,
                stroke,
                fill,
                evenodd,
                pts,
                transformed_path,
                passthrough_instruction,
                xobj_id,
                current_clip_paths,
                path,
            )
        elif shape in {"mlllh", "mllll"}:
            self._paint_rect_or_curve(
                gstate,
                stroke,
                fill,
                evenodd,
                pts,
                transformed_path,
                passthrough_instruction,
                xobj_id,
                current_clip_paths,
                path,
            )
        else:
            self.cur_item.add(
                self._make_curve(
                    gstate,
                    stroke,
                    fill,
                    evenodd,
                    pts,
                    transformed_path,
                    passthrough_instruction,
                    xobj_id,
                    current_clip_paths,
                    path,
                )
            )

    def _paint_valid_path(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        path: Sequence[PathSegment],
        current_clip_paths: object,
    ) -> None:
        """Handle a valid (m-starting) path: transform, normalise, and paint."""
        # Although the 'h' command does not not literally provide a
        # point-position, its position is (by definition) equal to the
        # subpath's starting point.
        #
        # And, per Section 4.4's Table 4.9, all other path commands place
        # their point-position in their final two arguments. (Any preceding
        # arguments represent control points on Bézier curves.)
        pts, transformed_path = self._transform_path(path)
        shape = "".join(x[0] for x in path)

        # Drop a redundant "l" on a path closed with "h"
        if len(shape) > 3 and shape[-2:] == "lh" and pts[-2] == pts[0]:
            shape = shape[:-2] + "h"
            pts.pop()

        passthrough_instruction = (
            self.il_creater.passthrough_per_char_instruction.copy()
        )
        xobj_id = self.il_creater.xobj_id
        self._dispatch_path_shape(
            gstate,
            stroke,
            fill,
            evenodd,
            path,
            shape,
            pts,
            transformed_path,
            passthrough_instruction,
            xobj_id,
            current_clip_paths,
        )

    def paint_path(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        path: Sequence[PathSegment],
    ) -> None:
        """Paint paths described in section 4.4 of the PDF reference manual"""
        shape = "".join(x[0] for x in path)
        current_clip_paths = self.il_creater.current_clip_paths.copy()
        if shape[:1] != "m":
            # Per PDF Reference Section 4.4.1, "path construction operators may
            # be invoked in any sequence, but the first one invoked must be m
            # or re to begin a new subpath." Since pdfminer.six already
            # converts all `re` (rectangle) operators to their equivelent
            # `mlllh` representation, paths ingested by `.paint_path(...)` that
            # do not begin with the `m` operator are invalid.
            return
        self._paint_valid_path(gstate, stroke, fill, evenodd, path, current_clip_paths)

    def render_char(
        self,
        matrix: Matrix,
        font: PDFFont,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs: PDFColorSpace,
        graphicstate: PDFGraphicState,
    ) -> float:
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        return item.adv

    def handle_undefined_char(self, font: PDFFont, cid: int) -> str:
        log.debug("undefined: %r, %r", font, cid)
        return "(cid:%d)" % cid

    def receive_layout(self, ltpage: LTPage) -> None:
        pass  # Intentionally not implemented: subclasses override this to process the analyzed page layout


class PDFPageAggregator(PDFLayoutAnalyzer):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        pageno: int = 1,
        laparams: LAParams | None = None,
    ) -> None:
        PDFLayoutAnalyzer.__init__(self, rsrcmgr, pageno=pageno, laparams=laparams)
        self.result: LTPage | None = None

    def receive_layout(self, ltpage: LTPage) -> None:
        self.result = ltpage

    def get_result(self) -> LTPage:
        assert self.result is not None
        return self.result


# Some PDFConverter children support only binary I/O
IOType = TypeVar("IOType", TextIO, BinaryIO, AnyIO)


class PDFConverter(PDFLayoutAnalyzer, Generic[IOType]):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        outfp: IOType,
        codec: str = "utf-8",
        pageno: int = 1,
        laparams: LAParams | None = None,
    ) -> None:
        PDFLayoutAnalyzer.__init__(self, rsrcmgr, pageno=pageno, laparams=laparams)
        self.outfp: IOType = outfp
        self.codec = codec
        self.outfp_binary = self._is_binary_stream(self.outfp)

    @staticmethod
    def _is_binary_stream(outfp: AnyIO) -> bool:
        """Test if an stream is binary or not"""
        if "b" in getattr(outfp, "mode", ""):
            return True
        elif hasattr(outfp, "mode"):
            # output stream has a mode, but it does not contain 'b'
            return False
        elif isinstance(outfp, io.BytesIO):
            return True
        elif isinstance(outfp, (io.StringIO, io.TextIOBase)):
            return False

        return True


class TextConverter(PDFConverter[AnyIO]):
    def __init__(  # NOSONAR - pdfminer-compatible converter API keeps these keyword parameters
        self,
        rsrcmgr: PDFResourceManager,
        outfp: AnyIO,
        codec: str = "utf-8",
        pageno: int = 1,
        laparams: LAParams | None = None,
        showpageno: bool = False,
        imagewriter: ImageWriter | None = None,
    ) -> None:
        super().__init__(rsrcmgr, outfp, codec=codec, pageno=pageno, laparams=laparams)
        self.showpageno = showpageno
        self.imagewriter = imagewriter

    def write_text(self, text: str) -> None:
        text = utils.compatible_encode_method(text, self.codec, "ignore")
        if self.outfp_binary:
            cast(BinaryIO, self.outfp).write(text.encode())
        else:
            cast(TextIO, self.outfp).write(text)

    def receive_layout(self, ltpage: LTPage) -> None:
        def render(item: LTItem) -> None:
            if isinstance(item, LTContainer):
                for child in item:
                    render(child)
            elif isinstance(item, LTText):
                self.write_text(item.get_text())
            if isinstance(item, LTTextBox):
                self.write_text("\n")
            elif isinstance(item, LTImage) and self.imagewriter is not None:
                self.imagewriter.export_image(item)

        if self.showpageno:
            self.write_text("Page %s\n" % ltpage.pageid)
        render(ltpage)
        self.write_text("\f")

    # Some dummy functions to save memory/CPU when all that is wanted
    # is text.  This stops all the image and drawing output from being
    # recorded and taking up RAM.
    def render_image(self, name: str, stream: PDFStream) -> None:
        if self.imagewriter is not None:
            PDFConverter.render_image(self, name, stream)

    def paint_path(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        path: Sequence[PathSegment],
    ) -> None:
        pass  # Intentionally not implemented: TextConverter ignores path drawing to save RAM


class HTMLConverter(PDFConverter[AnyIO]):
    RECT_COLORS = {
        "figure": "yellow",
        "textline": "magenta",
        "textbox": "cyan",
        "textgroup": "red",
        "curve": "black",
        "page": "gray",
    }

    TEXT_COLORS = {
        "textbox": "blue",
        "char": "black",
    }

    def __init__(  # NOSONAR - pdfminer-compatible converter API keeps these keyword parameters
        self,
        rsrcmgr: PDFResourceManager,
        outfp: AnyIO,
        codec: str = "utf-8",
        pageno: int = 1,
        laparams: LAParams | None = None,
        scale: float = 1,
        fontscale: float = 1.0,
        layoutmode: str = "normal",
        showpageno: bool = True,
        pagemargin: int = 50,
        imagewriter: ImageWriter | None = None,
        debug: int = 0,
        rect_colors_param: dict[str, str]
        | None = None,  # NOSONAR - renamed to avoid clash with RECT_COLORS class var
        text_colors_param: dict[str, str]
        | None = None,  # NOSONAR - renamed to avoid clash with TEXT_COLORS class var
    ) -> None:
        PDFConverter.__init__(
            self,
            rsrcmgr,
            outfp,
            codec=codec,
            pageno=pageno,
            laparams=laparams,
        )

        # write() assumes a codec for binary I/O, or no codec for text I/O.
        if self.outfp_binary and not self.codec:
            raise PDFValueError("Codec is required for a binary I/O output")
        if not self.outfp_binary and self.codec:
            raise PDFValueError("Codec must not be specified for a text I/O output")

        if text_colors_param is None:
            text_colors_param = {"char": "black"}
        if rect_colors_param is None:
            rect_colors_param = {"curve": "black", "page": "gray"}

        self.scale = scale
        self.fontscale = fontscale
        self.layoutmode = layoutmode
        self.showpageno = showpageno
        self.pagemargin = pagemargin
        self.imagewriter = imagewriter
        self.instance_rect_colors = rect_colors_param
        self.instance_text_colors = text_colors_param
        if debug:
            self.instance_rect_colors.update(self.RECT_COLORS)
            self.instance_text_colors.update(self.TEXT_COLORS)
        self._yoffset: float = self.pagemargin
        self._font: tuple[str, float] | None = None
        self._fontstack: list[tuple[str, float] | None] = []
        self.write_header()

    def write(self, text: str) -> None:
        if self.codec:
            cast(BinaryIO, self.outfp).write(text.encode(self.codec))
        else:
            cast(TextIO, self.outfp).write(text)

    def write_header(self) -> None:
        self.write("<html><head>\n")
        if self.codec:
            s = (
                '<meta http-equiv="Content-Type" content="text/html; '
                'charset=%s">\n' % self.codec
            )
        else:
            s = '<meta http-equiv="Content-Type" content="text/html">\n'
        self.write(s)
        self.write("</head><body>\n")

    def write_footer(self) -> None:
        page_links = [f'<a href="#{i}">{i}</a>' for i in range(1, self.pageno)]
        s = '<div style="position:absolute; top:0px;">Page: %s</div>\n' % ", ".join(
            page_links,
        )
        self.write(s)
        self.write("</body></html>\n")

    def write_text(self, text: str) -> None:
        self.write(enc(text))

    def place_rect(
        self,
        color: str,
        borderwidth: int,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> None:
        color2 = self.instance_rect_colors.get(color)
        if color2 is not None:
            s = (
                '<span style="position:absolute; border: %s %dpx solid; '
                'left:%dpx; top:%dpx; width:%dpx; height:%dpx;"></span>\n'
                % (
                    color2,
                    borderwidth,
                    x * self.scale,
                    (self._yoffset - y) * self.scale,
                    w * self.scale,
                    h * self.scale,
                )
            )
            self.write(s)

    def place_border(self, color: str, borderwidth: int, item: LTComponent) -> None:
        self.place_rect(color, borderwidth, item.x0, item.y1, item.width, item.height)

    def place_image(
        self,
        item: LTImage,
        borderwidth: int,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> None:
        if self.imagewriter is not None:
            name = self.imagewriter.export_image(item)
            s = (
                '<img src="%s" border="%d" style="position:absolute; '
                'left:%dpx; top:%dpx;" width="%d" height="%d" />\n'
                % (
                    enc(name),
                    borderwidth,
                    x * self.scale,
                    (self._yoffset - y) * self.scale,
                    w * self.scale,
                    h * self.scale,
                )
            )
            self.write(s)

    def place_text(
        self,
        color: str,
        text: str,
        x: float,
        y: float,
        size: float,
    ) -> None:
        color2 = self.instance_text_colors.get(color)
        if color2 is not None:
            s = (
                '<span style="position:absolute; color:%s; left:%dpx; '
                'top:%dpx; font-size:%dpx;">'
                % (
                    color2,
                    x * self.scale,
                    (self._yoffset - y) * self.scale,
                    size * self.scale * self.fontscale,
                )
            )
            self.write(s)
            self.write_text(text)
            self.write("</span>\n")

    def begin_div(
        self,
        color: str,
        borderwidth: int,
        x: float,
        y: float,
        w: float,
        h: float,
        writing_mode: str = "False",
    ) -> None:
        self._fontstack.append(self._font)
        self._font = None
        s = (
            '<div style="position:absolute; border: %s %dpx solid; '
            "writing-mode:%s; left:%dpx; top:%dpx; width:%dpx; "
            'height:%dpx;">'
            % (
                color,
                borderwidth,
                writing_mode,
                x * self.scale,
                (self._yoffset - y) * self.scale,
                w * self.scale,
                h * self.scale,
            )
        )
        self.write(s)

    def end_div(self, color: str) -> None:
        if self._font is not None:
            self.write("</span>")
        self._font = self._fontstack.pop()
        self.write("</div>")

    def put_text(self, text: str, fontname: str, fontsize: float) -> None:
        font = (fontname, fontsize)
        if font != self._font:
            if (
                self._font is not None
            ):  # NOSONAR - inner if is not the only statement in the outer block
                self.write("</span>")
            # Remove subset tag from fontname, see PDF Reference 5.5.3
            fontname_without_subset_tag = fontname.split("+")[-1]
            self.write(
                '<span style="font-family: %s; font-size:%dpx">'
                % (fontname_without_subset_tag, fontsize * self.scale * self.fontscale),
            )
            self._font = font
        self.write_text(text)

    def put_newline(self) -> None:
        self.write("<br>")

    def _show_html_group(self, item: "LTTextGroup | TextGroupElement") -> None:
        """Recursively render a text group hierarchy as HTML borders."""
        if isinstance(item, LTTextGroup):
            self.place_border("textgroup", 1, item)
            for child in item:
                self._show_html_group(child)

    def _render_html_item_exact(self, item: "LTItem") -> None:
        """Render one layout item in exact-mode positioning."""
        if isinstance(item, LTTextLine):
            self.place_border("textline", 1, item)
            for child in item:
                self._render_html_item(child)
        elif isinstance(item, LTTextBox):
            self.place_border("textbox", 1, item)
            self.place_text(
                "textbox",
                str(item.index + 1),
                item.x0,
                item.y1,
                20,
            )
            for child in item:
                self._render_html_item(child)
        elif isinstance(item, LTChar):
            self.place_border("char", 1, item)
            self.place_text(
                "char",
                item.get_text(),
                item.x0,
                item.y1,
                item.size,
            )

    def _render_html_page(self, item: "LTPage") -> None:
        """Render an LTPage node to HTML."""
        self._yoffset += item.y1
        self.place_border("page", 1, item)
        if self.showpageno:
            self.write(
                '<div style="position:absolute; top:%dpx;">'
                % ((self._yoffset - item.y1) * self.scale),
            )
            self.write(f'<a name="{item.pageid}">Page {item.pageid}</a></div>\n')
        for child in item:
            self._render_html_item(child)
        if item.groups is not None:
            for group in item.groups:
                self._show_html_group(group)

    def _render_html_text_line(self, item: "LTTextLine") -> None:
        """Render an LTTextLine item as HTML."""
        for child in item:
            self._render_html_item(child)
        if self.layoutmode != "loose":
            self.put_newline()

    def _render_html_text_box(self, item: "LTTextBox") -> None:
        """Render an LTTextBox item as HTML."""
        self.begin_div(
            "textbox",
            1,
            item.x0,
            item.y1,
            item.width,
            item.height,
            item.get_writing_mode(),
        )
        for child in item:
            self._render_html_item(child)
        self.end_div("textbox")

    def _render_html_item(  # NOSONAR - dispatcher must handle all LTItem subtypes; complexity is inherent
        self, item: "LTItem"
    ) -> None:
        """Recursively render a layout item and its children as HTML."""
        if isinstance(item, LTPage):
            self._render_html_page(item)
        elif isinstance(item, LTCurve):
            self.place_border("curve", 1, item)
        elif isinstance(item, LTFigure):
            self.begin_div("figure", 1, item.x0, item.y1, item.width, item.height)
            for child in item:
                self._render_html_item(child)
            self.end_div("figure")
        elif isinstance(item, LTImage):
            self.place_image(item, 1, item.x0, item.y1, item.width, item.height)
        elif self.layoutmode == "exact":
            self._render_html_item_exact(item)
        elif isinstance(item, LTTextLine):
            self._render_html_text_line(item)
        elif isinstance(item, LTTextBox):
            self._render_html_text_box(item)
        elif isinstance(item, LTChar):
            fontname = make_compat_str(item.fontname)
            self.put_text(item.get_text(), fontname, item.size)
        elif isinstance(item, LTText):
            self.write_text(item.get_text())

    def receive_layout(self, ltpage: LTPage) -> None:
        self._render_html_item(ltpage)
        self._yoffset += self.pagemargin

    def close(self) -> None:
        self.write_footer()


class XMLConverter(PDFConverter[AnyIO]):
    CONTROL = re.compile("[\x00-\x08\x0b-\x0c\x0e-\x1f]")

    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        outfp: AnyIO,
        codec: str = "utf-8",
        pageno: int = 1,
        laparams: LAParams | None = None,
        imagewriter: ImageWriter | None = None,
        stripcontrol: bool = False,
    ) -> None:
        PDFConverter.__init__(
            self,
            rsrcmgr,
            outfp,
            codec=codec,
            pageno=pageno,
            laparams=laparams,
        )

        # write() assumes a codec for binary I/O, or no codec for text I/O.
        if self.outfp_binary == (not self.codec):
            raise PDFValueError("Codec is required for a binary I/O output")

        self.imagewriter = imagewriter
        self.stripcontrol = stripcontrol
        self.write_header()

    def write(self, text: str) -> None:
        if self.codec:
            cast(BinaryIO, self.outfp).write(text.encode(self.codec))
        else:
            cast(TextIO, self.outfp).write(text)

    def write_header(self) -> None:
        if self.codec:
            self.write('<?xml version="1.0" encoding="%s" ?>\n' % self.codec)
        else:
            self.write('<?xml version="1.0" ?>\n')
        self.write("<pages>\n")

    def write_footer(self) -> None:
        self.write("</pages>\n")

    def write_text(self, text: str) -> None:
        if self.stripcontrol:
            text = self.CONTROL.sub("", text)
        self.write(enc(text))

    def _render_xml_image(self, item: "LTImage") -> None:
        """Write an XML image element to output."""
        if self.imagewriter is not None:
            name = self.imagewriter.export_image(item)
            self.write(
                '<image src="%s" width="%d" height="%d" />\n'
                % (enc(name), item.width, item.height),
            )
        else:
            self.write(
                '<image width="%d" height="%d" />\n' % (item.width, item.height),
            )

    def _render_xml_page(self, item: "LTPage", render) -> None:
        """Write XML for an LTPage node."""
        self.write(
            '<page id="%s" bbox="%s" rotate="%d">\n'
            % (item.pageid, bbox2str(item.bbox), item.rotate)
        )
        for child in item:
            render(child)
        if item.groups is not None:
            self.write("<layout>\n")
            for group in item.groups:
                self._show_group_xml(group)
            self.write("</layout>\n")
        self.write("</page>\n")

    def _render_xml_char(self, item: "LTChar") -> None:
        """Write XML for a single LTChar."""
        s = '<text font="%s" bbox="%s" colourspace="%s" ncolour="%s" size="%.3f">' % (
            enc(item.fontname),
            bbox2str(item.bbox),
            item.ncs.name,
            item.graphicstate.ncolor,
            item.size,
        )
        self.write(s)
        self.write_text(item.get_text())
        self.write("</text>\n")

    def _render_xml_shape(self, item: "LTItem") -> None:
        """Write XML for a primitive shape (line, rect, or curve)."""
        if isinstance(item, LTLine):
            self.write(
                '<line linewidth="%d" bbox="%s" />\n'
                % (item.linewidth, bbox2str(item.bbox))
            )
        elif isinstance(item, LTRect):
            self.write(
                '<rect linewidth="%d" bbox="%s" />\n'
                % (item.linewidth, bbox2str(item.bbox))
            )
        elif isinstance(item, LTCurve):
            self.write(
                '<curve linewidth="%d" bbox="%s" pts="%s"/>\n'
                % (item.linewidth, bbox2str(item.bbox), item.get_pts())
            )

    def _render_xml_text_containers(self, item: "LTItem", render: "Any") -> bool:
        """Write XML for text-container items. Returns True if handled."""
        if isinstance(item, LTTextLine):
            self.write('<textline bbox="%s">\n' % bbox2str(item.bbox))
            for child in item:
                render(child)
            self.write("</textline>\n")
            return True
        if isinstance(item, LTTextBox):
            wmode = ' wmode="vertical"' if isinstance(item, LTTextBoxVertical) else ""
            self.write(
                '<textbox id="%d" bbox="%s"%s>\n'
                % (item.index, bbox2str(item.bbox), wmode)
            )
            for child in item:
                render(child)
            self.write("</textbox>\n")
            return True
        return False

    def _render_xml_item(self, item: "LTItem", render: "Any") -> None:
        """Write one XML element for `item`, recursing into containers via `render`."""
        if isinstance(item, LTPage):
            self._render_xml_page(item, render)
        elif isinstance(item, (LTLine, LTRect, LTCurve)):
            self._render_xml_shape(item)
        elif isinstance(item, LTFigure):
            self.write(f'<figure name="{item.name}" bbox="{bbox2str(item.bbox)}">\n')
            for child in item:
                render(child)
            self.write("</figure>\n")
        elif self._render_xml_text_containers(item, render):
            return
        elif isinstance(item, LTChar):
            self._render_xml_char(item)
        elif isinstance(item, LTText):
            self.write("<text>%s</text>\n" % item.get_text())
        elif isinstance(item, LTImage):
            self._render_xml_image(item)
        else:
            assert False, str(("Unhandled", item))

    def _show_group_xml(self, item: "LTItem") -> None:
        """Recursively write XML for a text-group hierarchy."""
        if isinstance(item, LTTextBox):
            self.write(
                '<textbox id="%d" bbox="%s" />\n' % (item.index, bbox2str(item.bbox)),
            )
        elif isinstance(item, LTTextGroup):
            self.write('<textgroup bbox="%s">\n' % bbox2str(item.bbox))
            for child in item:
                self._show_group_xml(child)
            self.write("</textgroup>\n")

    def receive_layout(self, ltpage: LTPage) -> None:
        def render(item: LTItem) -> None:
            self._render_xml_item(item, render)

        render(ltpage)

    def close(self) -> None:
        self.write_footer()


class HOCRConverter(PDFConverter[AnyIO]):
    """Extract an hOCR representation from explicit text information within a PDF."""

    #   Where text is being extracted from a variety of types of PDF within a
    #   business process, those PDFs where the text is only present in image
    #   form will need to be analysed using an OCR tool which will typically
    #   output hOCR. This converter extracts the explicit text information from
    #   those PDFs that do have it and uses it to genxerate a basic hOCR
    #   representation that is designed to be used in conjunction with the image
    #   of the PDF in the same way as genuine OCR output would be, but without the
    #   inevitable OCR errors.

    #   The converter does not handle images, diagrams or text colors.

    #   In the examples processed by the contributor it was necessary to set
    #   LAParams.all_texts to True.

    CONTROL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")

    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        outfp: AnyIO,
        codec: str = "utf8",
        pageno: int = 1,
        laparams: LAParams | None = None,
        stripcontrol: bool = False,
    ):
        PDFConverter.__init__(
            self,
            rsrcmgr,
            outfp,
            codec=codec,
            pageno=pageno,
            laparams=laparams,
        )
        self.stripcontrol = stripcontrol
        self.within_chars = False
        self.write_header()

    def bbox_repr(self, bbox: Rect) -> str:
        (in_x0, in_y0, in_x1, in_y1) = bbox
        # PDF y-coordinates are the other way round from hOCR coordinates
        out_x0 = int(in_x0)
        out_y0 = int(self.page_bbox[3] - in_y1)
        out_x1 = int(in_x1)
        out_y1 = int(self.page_bbox[3] - in_y0)
        return f"bbox {out_x0} {out_y0} {out_x1} {out_y1}"

    def write(self, text: str) -> None:
        if self.codec:
            encoded_text = text.encode(self.codec)
            cast(BinaryIO, self.outfp).write(encoded_text)
        else:
            cast(TextIO, self.outfp).write(text)

    def write_header(self) -> None:
        if self.codec:
            self.write(
                "<html xmlns='http://www.w3.org/1999/xhtml' "
                "xml:lang='en' lang='en' charset='%s'>\n" % self.codec,
            )
        else:
            self.write(
                "<html xmlns='http://www.w3.org/1999/xhtml' xml:lang='en' lang='en'>\n",
            )
        self.write("<head>\n")
        self.write("<title></title>\n")
        self.write(
            "<meta http-equiv='Content-Type' content='text/html;charset=utf-8' />\n",
        )
        self.write(
            "<meta name='ocr-system' content='pdfminer.six HOCR Converter' />\n",
        )
        self.write(
            "  <meta name='ocr-capabilities'"
            " content='ocr_page ocr_block ocr_line ocrx_word'/>\n",
        )
        self.write("</head>\n")
        self.write("<body>\n")

    def write_footer(self) -> None:
        self.write("<!-- comment in the following line to debug -->\n")
        self.write(
            "<!--script src='https://unpkg.com/hocrjs'></script--></body></html>\n",
        )

    def write_text(self, text: str) -> None:
        if self.stripcontrol:
            text = self.CONTROL.sub("", text)
        self.write(text)

    def _build_font_styles(self) -> str:
        """Build CSS font style string based on working_font."""
        bold_and_italic_styles = ""
        if "Italic" in self.working_font:
            bold_and_italic_styles = "font-style: italic; "
        if "Bold" in self.working_font:
            bold_and_italic_styles += "font-weight: bold; "
        return bold_and_italic_styles

    def write_word(self) -> None:
        if len(self.working_text) > 0:
            bold_and_italic_styles = self._build_font_styles()
            self.write(
                "<span style='font:\"%s\"; font-size:%d; %s' "
                "class='ocrx_word' title='%s; x_font %s; "
                "x_fsize %d'>%s</span>"
                % (
                    self.working_font,
                    self.working_size,
                    bold_and_italic_styles,
                    self.bbox_repr(self.working_bbox),
                    self.working_font,
                    self.working_size,
                    self.working_text.strip(),
                ),
            )
        self.within_chars = False

    def _handle_hocr_char_continuation(self, item: "LTChar") -> None:
        """Handle an LTChar when already inside a word (within_chars=True)."""
        if len(item.get_text().strip()) == 0:
            self.write_word()
            self.write(item.get_text())
        else:
            if (
                self.working_bbox[1] != item.bbox[1]
                or self.working_font != item.fontname
                or self.working_size != item.size
            ):
                self.write_word()
                self.working_bbox = item.bbox
                self.working_font = item.fontname
                self.working_size = item.size
            self.working_text += item.get_text()
            self.working_bbox = (
                self.working_bbox[0],
                self.working_bbox[1],
                item.bbox[2],
                self.working_bbox[3],
            )

    def _handle_hocr_char(self, item: "LTChar") -> None:
        """Handle an LTChar item in the HOCR render pass."""
        if not self.within_chars:
            self.within_chars = True
            self.working_text = item.get_text()
            self.working_bbox = item.bbox
            self.working_font = item.fontname
            self.working_size = item.size
        else:
            self._handle_hocr_char_continuation(item)

    def _render_hocr_item(self, item: LTItem) -> None:
        """Recursively render a single layout item into HOCR output."""
        if self.within_chars and isinstance(item, LTAnno):
            self.write_word()
        if isinstance(item, LTPage):
            self.page_bbox = item.bbox
            self.write(
                "<div class='ocr_page' id='%s' title='%s'>\n"
                % (item.pageid, self.bbox_repr(item.bbox)),
            )
            for child in item:
                self._render_hocr_item(child)
            self.write("</div>\n")
        elif isinstance(item, LTTextLine):
            self.write(
                "<span class='ocr_line' title='%s'>" % self.bbox_repr(item.bbox),
            )
            for child_line in item:
                self._render_hocr_item(child_line)
            self.write("</span>\n")
        elif isinstance(item, LTTextBox):
            self.write(
                "<div class='ocr_block' id='%d' title='%s'>\n"
                % (item.index, self.bbox_repr(item.bbox)),
            )
            for child in item:
                self._render_hocr_item(child)
            self.write("</div>\n")
        elif isinstance(item, LTChar):
            self._handle_hocr_char(item)

    def receive_layout(self, ltpage: LTPage) -> None:
        self._render_hocr_item(ltpage)

    def close(self) -> None:
        self.write_footer()
