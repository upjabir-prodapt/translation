import logging

from pymupdf import Font

from src.doctranslator.format.pdf.document_il.frontend.il_creater import ILCreater
from src.doctranslator.pdfminer.converter import PDFConverter
from src.doctranslator.pdfminer.layout import LTChar
from src.doctranslator.pdfminer.layout import LTComponent
from src.doctranslator.pdfminer.layout import LTCurve
from src.doctranslator.pdfminer.layout import LTFigure
from src.doctranslator.pdfminer.layout import LTPage
from src.doctranslator.pdfminer.layout import LTText
from src.doctranslator.pdfminer.pdfcolor import PDFColorSpace
from src.doctranslator.pdfminer.pdffont import PDFFont
from src.doctranslator.pdfminer.pdffont import PDFUnicodeNotDefined
from src.doctranslator.pdfminer.pdfinterp import PDFGraphicState
from src.doctranslator.pdfminer.pdfinterp import PDFResourceManager
from src.doctranslator.pdfminer.utils import Matrix
from src.doctranslator.pdfminer.utils import apply_matrix_pt
from src.doctranslator.pdfminer.utils import bbox2str
from src.doctranslator.pdfminer.utils import matrix2str
from src.doctranslator.pdfminer.utils import mult_matrix

log = logging.getLogger(__name__)


class PDFConverterEx(PDFConverter):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        il_creater: ILCreater | None = None,
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)
        self.il_creater = il_creater

    def begin_page(self, page, ctm) -> None:
        # 重载替换 cropbox
        (x0, y0, x1, y1) = page.cropbox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.il_creater.on_page_media_box(
            mediabox[0],
            mediabox[1],
            mediabox[2],
            mediabox[3],
        )
        self.il_creater.on_page_number(page.pageno)
        self.cur_item = LTPage(page.pageno, mediabox)

    def end_page(self, _page) -> None:
        # 重载返回指令流
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        # 重载设置 pageid
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        # 重载返回指令流
        fig = self.cur_item
        if not isinstance(self.cur_item, LTFigure):
            raise ValueError(f"Unexpected item type: {type(self.cur_item)}")
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return self.receive_layout(fig)

    def render_char(
        self,
        matrix,
        font,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs,
        graphicstate: PDFGraphicState,
    ) -> float:
        # 重载设置 cid 和 font
        try:
            text = font.to_unichr(cid)
            if not isinstance(text, str):
                raise TypeError(f"Expected string, got {type(text)}")
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        font_id = font.font_id_temp
        if font_id is None:
            if not hasattr(font, "xobj_id"):
                log.debug(
                    f"Font {font.fontname} does not have xobj_id attribute.",
                )
                font_id = "UNKNOW"
            else:
                font_id = self.il_creater.current_page_font_name_id_map.get(
                    font.xobj_id, None
                )

        item = AWLTChar(
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
            self.il_creater.xobj_id,
            font_id,
            self.il_creater.get_render_order_and_increase(),
        )
        self.cur_item.add(item)
        item.cid = cid  # hack 插入原字符编码
        item.font = font  # hack 插入原字符字体
        return item.adv


class AWLTChar(LTChar):
    """Actual letter in the text as a Unicode string."""

    def __init__(
        self,
        matrix: Matrix,
        font: PDFFont,
        fontsize: float,
        scaling: float,
        rise: float,
        text: str,
        textwidth: float,
        textdisp: float | tuple[float | None, float],
        ncs: PDFColorSpace,
        graphicstate: PDFGraphicState,
        xobj_id: int,
        font_id: str,
        render_order: int,
    ) -> None:
        LTText.__init__(self)
        self._text = text
        self.matrix = matrix
        self.fontname = font.fontname
        self.ncs = ncs
        self.graphicstate = graphicstate
        self.xobj_id = xobj_id
        self.adv = textwidth * fontsize * scaling
        self.aw_font_id = font_id
        self.render_order = render_order
        # compute the boundary rectangle.
        if font.is_vertical():
            # vertical
            if not isinstance(textdisp, tuple):
                raise AssertionError
            (vx, vy) = textdisp
            if vx is None:
                vx = fontsize * 0.5
            else:
                vx = vx * fontsize * 0.001
            vy = (1000 - vy) * fontsize * 0.001
            bbox_lower_left = (-vx, vy + rise + self.adv)
            bbox_upper_right = (-vx + fontsize, vy + rise)
        else:
            # horizontal
            descent = font.get_descent() * fontsize
            bbox_lower_left = (0, descent + rise)
            bbox_upper_right = (self.adv, descent + rise + fontsize)
        (a, b, c, d, e, f) = self.matrix
        self.upright = a * d * scaling > 0 and b * c <= 0
        (x0, y0) = apply_matrix_pt(self.matrix, bbox_lower_left)
        (x1, y1) = apply_matrix_pt(self.matrix, bbox_upper_right)
        if x1 < x0:
            (x0, x1) = (x1, x0)
        if y1 < y0:
            (y0, y1) = (y1, y0)
        LTComponent.__init__(self, (x0, y0, x1, y1))
        if font.is_vertical() or matrix[0] == 0:
            self.size = self.width
        else:
            self.size = self.height

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {bbox2str(self.bbox)} matrix={matrix2str(self.matrix)} font={self.fontname!r} adv={self.adv} text={self.get_text()!r}>"

    def get_text(self) -> str:
        return self._text


# fmt: off
class TranslateConverter(PDFConverterEx):
    def __init__(
        self,
        rsrcmgr,
        vfont: str | None = None,
        vchar: str | None = None,
        thread: int = 0,
        layout: dict | None = None,
        lang_in: str = "",  # 保留参数但添加未使用标记
        _lang_out: str = "",  # 改为未使用参数
        _service: str = "",  # 改为未使用参数
        resfont: str = "",
        noto: Font | None = None,
        envs: dict | None = None,
        _prompt: list | None = None,  # 改为未使用参数
        il_creater: ILCreater | None = None,
    ):
        layout = layout or {}
        super().__init__(rsrcmgr, il_creater)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        self.resfont = resfont
        self.noto = noto

    def receive_layout(self, ltpage: LTPage) -> None:
        """Process a PDF page layout and forward each element to the IL creator.

        Each LTChar is forwarded to ILCreater.on_lt_char for intermediate-language
        construction.  LTFigure and LTCurve objects are forwarded to their
        respective handlers.  All other layout elements are intentionally ignored.
        """
        for child in ltpage:
            if isinstance(child, LTChar):
                try:
                    self.il_creater.on_lt_char(child)
                except Exception:
                    log.exception("Error processing LTChar")
            elif isinstance(child, LTFigure):
                self.il_creater.on_pdf_figure(child)
            elif isinstance(child, LTCurve):
                self.il_creater.on_lt_curve(child)
