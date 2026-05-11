import logging
from collections.abc import Sequence
from typing import Any
from typing import cast

import numpy as np

from src.doctranslator.format.pdf.babelpdf.utils import guarded_bbox
from src.doctranslator.format.pdf.document_il.frontend.il_creater import ILCreater
from src.doctranslator.pdfminer import settings
from src.doctranslator.pdfminer.pdfcolor import PREDEFINED_COLORSPACE
from src.doctranslator.pdfminer.pdfcolor import PDFColorSpace
from src.doctranslator.pdfminer.pdfdevice import PDFDevice
from src.doctranslator.pdfminer.pdfdevice import PDFTextSeq
from src.doctranslator.pdfminer.pdffont import PDFFont
from src.doctranslator.pdfminer.pdfinterp import LITERAL_FORM
from src.doctranslator.pdfminer.pdfinterp import LITERAL_IMAGE
from src.doctranslator.pdfminer.pdfinterp import Color
from src.doctranslator.pdfminer.pdfinterp import PDFContentParser
from src.doctranslator.pdfminer.pdfinterp import PDFInterpreterError
from src.doctranslator.pdfminer.pdfinterp import PDFPageInterpreter
from src.doctranslator.pdfminer.pdfinterp import PDFResourceManager
from src.doctranslator.pdfminer.pdfinterp import PDFStackT
from src.doctranslator.pdfminer.pdfpage import PDFPage
from src.doctranslator.pdfminer.pdftypes import LITERALS_ASCII85_DECODE
from src.doctranslator.pdfminer.pdftypes import PDFObjRef
from src.doctranslator.pdfminer.pdftypes import PDFStream
from src.doctranslator.pdfminer.pdftypes import dict_value
from src.doctranslator.pdfminer.pdftypes import list_value
from src.doctranslator.pdfminer.pdftypes import resolve1
from src.doctranslator.pdfminer.pdftypes import stream_value
from src.doctranslator.pdfminer.psexceptions import PSEOF
from src.doctranslator.pdfminer.psexceptions import PSTypeError
from src.doctranslator.pdfminer.psparser import PSKeyword
from src.doctranslator.pdfminer.psparser import PSLiteral
from src.doctranslator.pdfminer.psparser import keyword_name
from src.doctranslator.pdfminer.psparser import literal_name
from src.doctranslator.pdfminer.utils import MATRIX_IDENTITY
from src.doctranslator.pdfminer.utils import Matrix
from src.doctranslator.pdfminer.utils import Rect
from src.doctranslator.pdfminer.utils import apply_matrix_pt
from src.doctranslator.pdfminer.utils import choplist
from src.doctranslator.pdfminer.utils import mult_matrix

log = logging.getLogger(__name__)


def safe_float(o: Any) -> float | None:
    try:
        return float(o)
    except (TypeError, ValueError):
        return None


class PDFContentParserEx(PDFContentParser):
    def __init__(self, streams: Sequence[object]) -> None:
        super().__init__(streams)

    def _handle_keyword_id(self, pos: int) -> None:
        """Handle the ID keyword for inline images."""
        try:
            (_, objs) = self.end_type("inline")
            if len(objs) % 2 != 0:
                error_msg = f"Invalid dictionary construct: {objs!r}"
                raise PSTypeError(error_msg)
            d = {literal_name(k): resolve1(v) for (k, v) in choplist(2, objs)}
            eos = b"EI"
            filter_ = d.get("F", None)
            if filter_:
                if isinstance(filter_, PSLiteral):
                    filter_ = [filter_]
                if filter_[0] in LITERALS_ASCII85_DECODE:
                    eos = b"~>"
            (pos, data) = self.get_inline_data(pos + len(b"ID "), target=eos)
            if eos != b"EI":  # it may be necessary for decoding
                data += eos
            obj = PDFStream(d, data)
            self.push((pos, obj))
            if eos == b"EI":  # otherwise it is still in the stream
                self.push((pos, self.KEYWORD_EI))
        except PSTypeError:
            if settings.STRICT:
                raise

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        if token is self.KEYWORD_BI:
            # inline image within a content stream
            self.start_type(pos, "inline")
        elif token is self.KEYWORD_ID:
            self._handle_keyword_id(pos)
        else:
            self.push((pos, token))


class PDFPageInterpreterEx(PDFPageInterpreter):
    """Processor for the content of a PDF page

    Reference: PDF Reference, Appendix A, Operator Summary
    """

    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        device: PDFDevice,
        obj_patch,
        il_creater: ILCreater,
    ) -> None:
        self.rsrcmgr = rsrcmgr
        self.device = device
        self.obj_patch = obj_patch
        self.il_creater = il_creater

    def dup(self) -> "PDFPageInterpreterEx":
        return self.__class__(
            self.rsrcmgr,
            self.device,
            self.obj_patch,
            self.il_creater,
        )

    @staticmethod
    def _get_colorspace(spec: object) -> PDFColorSpace | None:
        """Resolve a colorspace spec to a PDFColorSpace object."""
        if isinstance(spec, list):
            name = literal_name(spec[0])
        else:
            name = literal_name(spec)
        if name == "ICCBased" and isinstance(spec, list) and len(spec) >= 2:
            val = stream_value(spec[1])
            if "N" in val:
                return PDFColorSpace(name, val["N"])
            if "Alternate" in val:
                return PREDEFINED_COLORSPACE[val["Alternate"].name]
            return None
        if name == "DeviceN" and isinstance(spec, list) and len(spec) >= 2:
            return PDFColorSpace(name, len(list_value(spec[1])))
        return PREDEFINED_COLORSPACE.get(name)

    def _load_font_resources(self, v: object) -> None:
        """Populate fontmap and fontid from a Font resource dict."""
        for fontid, spec in dict_value(v).items():
            objid = None
            if isinstance(spec, PDFObjRef):
                objid = spec.objid
            spec = dict_value(spec)
            font = self.rsrcmgr.get_font(objid, spec)
            font.xobj_id = objid
            self.il_creater.on_page_resource_font(font, objid, fontid)
            self.fontmap[fontid] = font
            self.fontmap[fontid].descent = 0  # hack fix descent
            self.fontid[self.fontmap[fontid]] = fontid

    def _load_colorspace_resources(self, v: object) -> None:
        """Populate csmap from a ColorSpace resource dict."""
        for csid, spec in dict_value(v).items():
            colorspace = self._get_colorspace(resolve1(spec))
            if colorspace is not None:
                self.csmap[csid] = colorspace

    def init_resources(self, resources: dict[object, object]) -> None:
        # 重载设置 fontid 和 descent
        """Prepare the fonts and XObjects listed in the Resource attribute."""
        self.resources = resources
        self.fontmap: dict[object, PDFFont] = {}
        self.fontid: dict[PDFFont, object] = {}
        self.xobjmap = {}
        self.csmap: dict[str, PDFColorSpace] = PREDEFINED_COLORSPACE.copy()
        if not resources:
            return

        for k, v in dict_value(resources).items():
            if k == "Font":
                self._load_font_resources(v)
            elif k == "ColorSpace":
                self._load_colorspace_resources(v)
            elif k == "ProcSet":
                self.rsrcmgr.get_procset(list_value(v))
            elif k == "XObject":
                for xobjid, xobjstrm in dict_value(v).items():
                    self.xobjmap[xobjid] = xobjstrm

    def do_CS(self, name: PDFStackT) -> None:
        """Set color space for stroking operations

        Introduced in PDF 1.1
        """
        try:
            self.il_creater.on_stroking_color_space(literal_name(name))
            self.scs = self.csmap[literal_name(name)]
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError(f"Undefined ColorSpace: {name!r}") from None

    def do_cs_op(self, name: PDFStackT) -> None:
        """Set color space for nonstroking operations"""
        try:
            self.il_creater.on_non_stroking_color_space(literal_name(name))
            self.ncs = self.csmap[literal_name(name)]
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError(f"Undefined ColorSpace: {name!r}") from None

    # Keep the original name as an alias so existing callers continue to work
    do_cs = do_cs_op

    ############################################################
    # 重载返回调用参数（SCN）
    def do_SCN(self) -> None:
        """Set color for stroking operations."""
        if not self.scs:
            if settings.STRICT:
                raise PDFInterpreterError("No colorspace specified!")
        n = len(self.argstack)
        args = self.pop(n)
        self.il_creater.on_passthrough_per_char("SCN", args)
        self.graphicstate.scolor = cast(Color, args)
        return args

    def do_scn_op(self) -> None:
        """Set color for nonstroking operations"""
        if not self.ncs:
            if settings.STRICT:
                raise PDFInterpreterError("No colorspace specified!")
        n = len(self.argstack)
        args = self.pop(n)
        self.il_creater.on_passthrough_per_char("scn", args)
        self.graphicstate.ncolor = cast(Color, args)
        return args

    # Keep the original name as an alias so existing callers continue to work
    do_scn = do_scn_op

    def do_SC(self) -> None:
        """Set color for stroking operations"""
        args = self.do_SCN()
        self.il_creater.remove_latest_passthrough_per_char_instruction()
        self.il_creater.on_passthrough_per_char("SC", args)
        return args

    def do_sc_op(self) -> None:
        """Set color for nonstroking operations"""
        args = self.do_scn_op()
        self.il_creater.remove_latest_passthrough_per_char_instruction()
        self.il_creater.on_passthrough_per_char("sc", args)
        return args

    # Keep the original name as an alias so existing callers continue to work
    do_sc = do_sc_op

    @staticmethod
    def _compute_ctm_inverse(ctm) -> tuple | None:
        """Compute CTM inverse matrix components; returns None if singular."""
        try:
            ctm_inv = np.linalg.inv(np.array(ctm[:4]).reshape(2, 2))
        except Exception:
            return None
        np_version = np.__version__
        if np_version.split(".")[0] >= "2":
            pos_inv = -np.asmatrix(ctm[4:]) * ctm_inv
        else:
            pos_inv = -np.mat(ctm[4:]) * ctm_inv
        a, b, c, d = ctm_inv.reshape(4).tolist()
        e, f = pos_inv.tolist()[0]
        return a, b, c, d, e, f

    def _handle_form_xobj(self, xobj, xobjid: str) -> None:
        """Process a Form-type XObject."""
        # In extremely rare cases, a none might be mixed in the bbox, for example
        # /BBox [ 0 3.052 null 274.9 157.3 ]
        bbox = list(
            filter(lambda x: x is not None, cast(Rect, list_value(xobj["BBox"])))
        )
        if len(bbox) < 4:
            return

        matrix = cast(Matrix, list_value(xobj.get("Matrix", MATRIX_IDENTITY)))
        # According to PDF reference 1.7 section 4.9.1, XObjects in
        # earlier PDFs (prior to v1.2) use the page's Resources entry
        # instead of having their own Resources entry.
        xobjres = xobj.get("Resources")
        resources = dict_value(xobjres) if xobjres else self.resources.copy()

        self.il_creater.on_xobj_form(
            self.ctm,
            self.il_creater.xobj_id,
            xobj.objid,
            "form",
            xobjid,
            bbox,
            matrix,
        )

        self.device.begin_figure(xobjid, bbox, matrix)
        ctm = mult_matrix(matrix, self.ctm)
        (x, y, x2, y2) = guarded_bbox(bbox)
        (x, y) = apply_matrix_pt(ctm, (x, y))
        (x2, y2) = apply_matrix_pt(ctm, (x2, y2))
        x_id = self.il_creater.on_xobj_begin((x, y, x2, y2), xobj.objid)
        inv = self._compute_ctm_inverse(ctm)
        if inv is None:
            self.il_creater.on_xobj_end(x_id, " ")
            return
        a, b, c, d, e, f = inv
        interpreter = self.dup()
        ops_base = interpreter.render_contents(resources, [xobj], ctm=ctm)
        self.ncs = interpreter.ncs
        self.scs = interpreter.scs
        self.il_creater.on_xobj_end(
            x_id,
            f"{a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.6f} {f:.6f} cm ",
        )
        try:  # 有的时候 form 字体加不上这里会烂掉
            self.device.fontid = interpreter.fontid
            self.device.fontmap = interpreter.fontmap
            ops_new = self.device.end_figure(xobjid)
            inv2 = self._compute_ctm_inverse(ctm)
            if inv2 is not None:
                a, b, c, d, e, f = inv2
            self.obj_patch[self.xobjmap[xobjid].objid] = (
                f"q {ops_base}Q {a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.6f} {f:.6f} cm {ops_new}"
            )
        except Exception:
            pass

    def _handle_image_xobj(self, xobj, xobjid: str) -> None:
        """Process an Image-type XObject."""
        self.il_creater.on_xobj_form(
            self.ctm,
            self.il_creater.xobj_id,
            xobj.objid,
            "image",
            xobjid,
            (0, 0, 1, 1),
            MATRIX_IDENTITY,
        )
        self.device.begin_figure(xobjid, (0, 0, 1, 1), MATRIX_IDENTITY)
        self.device.render_image(xobjid, xobj)
        self.device.end_figure(xobjid)

    # Ensure bbox has four numbers, otherwise determine it as an illegal image
    # For example, some Form's bbox is '[ null -.00487 1.00412 .99393 ]'
    def do_Do(self, xobjid_arg: PDFStackT) -> None:
        # 重载设置 xobj 的 obj_patch
        """Invoke named XObject"""
        xobjid = literal_name(xobjid_arg)
        try:
            xobj = stream_value(self.xobjmap[xobjid])
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError(f"Undefined xobject id: {xobjid!r}") from None
            return
        subtype = xobj.get("Subtype")
        if subtype is LITERAL_FORM and "BBox" in xobj:
            self._handle_form_xobj(xobj, xobjid)
        elif subtype is LITERAL_IMAGE and "Width" in xobj and "Height" in xobj:
            self._handle_image_xobj(xobj, xobjid)

    def do_W(self) -> None:
        """Set clipping path using nonzero winding number rule"""
        self.handle_w(False)

    def do_W_a(self) -> None:
        """Set clipping path using even-odd rule"""
        self.handle_w(True)

    def handle_w(self, evenodd: bool):
        path = self.curpath
        self.il_creater.on_pdf_clip_path(path, evenodd, self.ctm)

    def process_page(self, page: PDFPage) -> str:
        # 重载设置 page 的 obj_patch
        (x0, y0, x1, y1) = page.cropbox
        ctm_for_ops = (1, 0, 0, 1, -x0, -y0)
        ctm = (1, 0, 0, 1, -x0, -y0)
        if page.rotate == 90 or page.rotate == 270:
            (x0, y0, x1, y1) = (y0, x1, y1, x0)
        self.il_creater.on_page_start()
        self.il_creater.on_page_crop_box(x0, y0, x1, y1)
        self.device.begin_page(page, ctm)
        ops_base = self.render_contents(page.resources, page.contents, ctm=ctm)
        self.device.fontid = self.fontid
        self.device.fontmap = self.fontmap
        _ops_new = self.device.end_page(page)
        return f"q {ops_base} Q {' '.join(f'{x:f}' for x in ctm_for_ops)} cm"

    def render_contents(
        self,
        resources: dict[object, object],
        streams: Sequence[object],
        ctm: Matrix = MATRIX_IDENTITY,
    ) -> None:
        # 重载返回指令流
        """Render the content streams.

        This method may be called recursively.
        """
        # log.debug(
        #     "render_contents: resources=%r, streams=%r, ctm=%r",
        #     resources,
        #     streams,
        #     ctm,
        # )
        self.init_resources(resources)
        self.init_state(ctm)
        return self.execute(list_value(streams))

    def do_q(self) -> None:
        """Save graphics state"""
        self.gstack.append(self.get_current_state())
        self.il_creater.push_passthrough_per_char_instruction()

    def do_Q_op(self) -> None:
        """Restore graphics state"""
        if self.gstack:
            self.set_current_state(self.gstack.pop())
        self.il_creater.pop_passthrough_per_char_instruction()

    # Keep the original name as an alias so existing callers continue to work
    do_Q = do_Q_op

    def do_TJ(self, seq: PDFStackT) -> None:
        """Show text, allowing individual glyph positioning"""
        if self.textstate.font is None:
            if settings.STRICT:
                raise PDFInterpreterError("No font specified!")
            return
        if isinstance(seq, PSLiteral):
            return
        assert self.ncs is not None
        gs = self.graphicstate.copy()
        gs.passthrough_instruction = (
            self.il_creater.passthrough_per_char_instruction.copy()
        )
        if isinstance(seq, int) or isinstance(seq, float):
            seq = [seq]
        self.device.render_string(self.textstate, cast(PDFTextSeq, seq), self.ncs, gs)

    def do_d(self, dash: PDFStackT, phase: PDFStackT) -> None:
        """Set line dash pattern"""
        self.graphicstate.dash = (dash, phase)
        self.il_creater.on_line_dash(dash, phase)

    def do_BI(self) -> None:
        """Begin inline image object"""
        self.il_creater.on_inline_image_begin()

    def do_ID(self) -> None:
        """Begin inline image data"""
        pass  # Handled by PDFContentParserEx

    def do_EI(self, obj: PDFStackT) -> None:
        """End inline image object"""
        if isinstance(obj, PDFStream):
            self.il_creater.on_inline_image_end(obj, self.ctm)

    @staticmethod
    def _format_args(args) -> str:
        """Format a list of arguments as a PDF operand string."""
        return " ".join(
            f"{x:f}" if isinstance(x, float) else str(x).replace("'", "")
            for x in args
        )

    def _build_ops_for_args_op(self, name: str, args) -> str:
        """Return the ops string contribution for an operator that takes arguments."""
        if self.il_creater.is_graphic_operation(name):
            return ""
        if name == "d":
            arg0 = f"[{' '.join(f'{arg}' for arg in args[0])}]"
            arg1 = args[1]
            return f"{arg0} {arg1} {name} "
        # 过滤 T 系列文字指令，EI，和 marked 系列指令
        if name[0] == "T" or name in ['"', "'", "EI", "MP", "DP", "BMC", "BDC"]:
            return ""
        return f"{self._format_args(args)} {name} "

    def _build_ops_for_no_args_op(self, name: str, targs) -> str:
        """Return the ops string contribution for a zero-argument operator."""
        if self.il_creater.is_graphic_operation(name):
            return ""
        if name[0] == "T" or name in ["BI", "ID", "EMC"]:
            return ""
        return f"{self._format_args(targs)} {name} "

    def _dispatch_keyword(self, name: str) -> tuple[bool, str]:
        """Dispatch a single PDF keyword; return (should_continue, ops_fragment)."""
        act_name = name.replace("*", "_a").replace('"', "_w").replace("'", "_q")
        method = f"do_{act_name}"
        if not hasattr(self, method):
            if settings.STRICT:
                raise PDFInterpreterError(f"Unknown operator: {name!r}")
            return False, ""

        func = getattr(self, method)
        nargs = func.__code__.co_argcount - 1
        if nargs:
            args = self.pop(nargs)
            if len(args) != nargs:
                return False, ""
            func(*args)
            if self.il_creater.is_passthrough_per_char_operation(name):
                self.il_creater.on_passthrough_per_char(name, args)
            ops_fragment = self._build_ops_for_args_op(name, args)
            should_continue = self.il_creater.is_graphic_operation(name)
        else:
            targs = func()
            if targs is None:
                targs = []
            ops_fragment = self._build_ops_for_no_args_op(name, targs)
            should_continue = self.il_creater.is_graphic_operation(name)

        return should_continue, ops_fragment

    # Run PostScript commands
    # The Do_xxx method is the method for executing corresponding postscript instructions
    def execute(self, streams: Sequence[object]) -> str:
        ops = ""
        for stream in streams:
            self.il_creater.on_new_stream()
            # 重载返回指令流
            try:
                parser = PDFContentParserEx([stream])
            except PSEOF:
                # empty page
                return ops
            while True:
                try:
                    (_, obj) = parser.nextobject()
                except PSEOF:
                    break
                if isinstance(obj, PSKeyword):
                    name = keyword_name(obj)
                    should_continue, ops_fragment = self._dispatch_keyword(name)
                    if should_continue:
                        continue
                    ops += ops_fragment
                else:
                    self.push(obj)
        return ops
