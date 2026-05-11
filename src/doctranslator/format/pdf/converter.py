import logging
import re
import unicodedata

import numpy as np
from pymupdf import Font

from src.doctranslator.format.pdf.document_il.frontend.il_creater import ILCreater
from src.doctranslator.pdfminer.converter import PDFConverter
from src.doctranslator.pdfminer.layout import LTChar
from src.doctranslator.pdfminer.layout import LTComponent
from src.doctranslator.pdfminer.layout import LTCurve
from src.doctranslator.pdfminer.layout import LTFigure
from src.doctranslator.pdfminer.layout import LTLine
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
            assert isinstance(textdisp, tuple)
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


class Paragraph:
    def __init__(self, y, x, x0, x1, size, brk):
        self.y: float = y  # 初始纵坐标
        self.x: float = x  # 初始横坐标
        self.x0: float = x0  # 左边界
        self.x1: float = x1  # 右边界
        self.size: float = size  # 字体大小
        self.brk: bool = brk  # 换行标记


# Latex font name patterns used to identify formula characters
_LATEX_FONT_PATTERNS = [
    r"CM[^R]",
    r"(MS|XY|MT|BL|RM|EU|LA|RS)[A-Z]",
    r"LINE",
    r"LCIRCLE",
    r"TeX-",
    r"rsfs",
    r"txsy",
    r"wasy",
    r"stmary",
    r".*Mono",
    r".*Code",
    r".*Ital",
    r".*Sym",
    r".*Math",
]


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

    def receive_layout(self, ltpage: LTPage):
        # 段落
        sstk: list[str] = []            # 段落文字栈
        pstk: list[Paragraph] = []      # 段落属性栈
        vbkt: int = 0                   # 段落公式括号计数
        # 公式组
        vstk: list[LTChar] = []         # 公式符号组
        vlstk: list[LTLine] = []        # 公式线条组
        vfix: float = 0                 # 公式纵向偏移
        # 公式组栈
        var: list[list[LTChar]] = []    # 公式符号组栈
        varl: list[list[LTLine]] = []   # 公式线条组栈
        varf: list[float] = []          # 公式纵向偏移栈
        vlen: list[float] = []          # 公式宽度栈
        # 全局
        lstk: list[LTLine] = []         # 全局线条栈
        xt: LTChar | None = None        # 上一个字符
        xt_cls: int = -1                # 上一个字符所属段落，保证无论第一个字符属于哪个类别都可以触发新段落
        vmax: float = ltpage.width / 4  # 行内公式最大宽度
        ops: str = ""                   # 渲染结果

        def vflag(font: str, char: str):    # 匹配公式（和角标）字体
            if isinstance(font, bytes):     # 不一定能 decode，直接转 str
                font = str(font)
            font = font.split("+")[-1]      # 字体名截断
            if re.match(r"\(cid:", char):
                return True
            # 基于字体名规则的判定
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                # latex 字体 - use module-level constant list of simple patterns
                if any(re.match(p, font) for p in _LATEX_FONT_PATTERNS):
                    return True
            # 基于字符集规则的判定
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                if (
                    char
                    and char != " "                                     # 非空格
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]   # 文字修饰符、数学符号、分隔符号
                        or ord(char[0]) in range(0x370, 0x400)          # 希腊字母
                    )
                ):
                    return True
            return False

        ############################################################
        # A. 原文档解析
        for child in ltpage:
            if isinstance(child, LTChar):
                try:
                    self.il_creater.on_lt_char(child)
                except Exception:
                    log.exception(
                        'Error processing LTChar',
                    )
                continue
                cur_v = False
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前字符在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                # 锚定文档中 bullet 的位置
                if child.get_text() == "•":
                    cls = 0
                # 判定当前字符是否属于公式
                if (                                                                                        # 判定当前字符是否属于公式
                    cls == 0                                                                                # 1. 类别为保留区域
                    or (cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79)  # 2. 角标字体，有 0.76 的角标和 0.799 的大写，这里用 0.79 取中，同时考虑首字母放大的情况
                    or vflag(child.fontname, child.get_text())                                              # 3. 公式字体
                    or (child.matrix[0] == 0 and child.matrix[3] == 0)                                      # 4. 垂直字体
                ):
                    cur_v = True
                # 判定括号组是否属于公式
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (                                                        # 判定当前公式是否结束
                    not cur_v                                               # 1. 当前字符不属于公式
                    or cls != xt_cls                                        # 2. 当前字符与前一个字符不属于同一段落
                    # or (abs(child.x0 - xt.x0) > vmax and cls != 0)        # 3. 段落内换行，可能是一长串斜体的段落，也可能是段内分式换行，这里设个阈值进行区分
                    # 禁止纯公式（代码）段落换行，直到文字开始再重开文字段落，保证只存在两种情况
                    # A. 纯公式（代码）段落（锚定绝对位置）sstk[-1]=="" -> sstk[-1]=="{v*}"
                    # B. 文字开头段落（排版相对位置）sstk[-1]!=""
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)    # 因为 cls==xt_cls==0 一定有 sstk[-1]==""，所以这里不需要再判定 cls!=0
                ):
                    if vstk:
                        if (                                                # 根据公式右侧的文字修正公式的纵向偏移
                            not cur_v                                       # 1. 当前字符不属于公式
                            and cls == xt_cls                               # 2. 当前字符与前一个字符属于同一段落
                            and child.x0 > max([vch.x0 for vch in vstk])    # 3. 当前字符在公式右侧
                        ):
                            vfix = vstk[0].y0 - child.y0
                        if sstk[-1] == "":
                            xt_cls = -1 # 禁止纯公式段落（sstk[-1]=="{v*}"）的后续连接，但是要考虑新字符和后续字符的连接，所以这里修改的是上个字符的类别
                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                # 当前字符不属于公式或当前字符是公式的第一个字符
                if not vstk:
                    if cls == xt_cls:               # 当前字符与前一个字符属于同一段落
                        if child.x0 > xt.x1 + 1:    # 添加行内空格
                            sstk[-1] += " "
                        elif child.x1 < xt.x0:      # 添加换行空格并标记原文段落存在换行
                            sstk[-1] += " "
                            pstk[-1].brk = True
                    else:                           # 根据当前字符构建一个新的段落
                        sstk.append("")
                        pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.size, False))
                if not cur_v:                                               # 文字入栈
                    if (                                                    # 根据当前字符修正段落属性
                        child.size > pstk[-1].size / 0.79                   # 1. 当前字符显著比段落字体大
                        or len(sstk[-1].strip()) == 1                       # 2. 当前字符为段落第二个文字（考虑首字母放大的情况）
                    ) and child.get_text() != " ":                          # 3. 当前字符不是空格
                        pstk[-1].y -= child.size - pstk[-1].size            # 修正段落初始纵坐标，假设两个不同大小字符的上边界对齐
                        pstk[-1].size = child.size
                    sstk[-1] += child.get_text()
                else:                                                       # 公式入栈
                    if (                                                    # 根据公式左侧的文字修正公式的纵向偏移
                        not vstk                                            # 1. 当前字符是公式的第一个字符
                        and cls == xt_cls                                   # 2. 当前字符与前一个字符属于同一段落
                        and child.x0 > xt.x0                                # 3. 前一个字符在公式左侧
                    ):
                        vfix = child.y0 - xt.y0
                    vstk.append(child)
                # 更新段落边界，因为段落内换行之后可能是公式开头，所以要在外边处理
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                # 更新上一个字符
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):
                # 图表
                self.il_creater.on_pdf_figure(child)
            # elif isinstance(child, LTLine):     # 线条
            #     continue
            #     layout = self.layout[ltpage.pageid]
            #     # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
            #     h, w = layout.shape
            #     # 读取当前线条在 layout 中的类别
            #     cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
            #     cls = layout[cy, cx]
            #     if vstk and cls == xt_cls:      # 公式线条
            #         vlstk.append(child)
            #     else:                           # 全局线条
            #         lstk.append(child)
            elif isinstance(child, LTCurve):
                self.il_creater.on_lt_curve(child)
        return ops
