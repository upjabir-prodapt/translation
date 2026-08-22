from src.worker.doctranslator.pdfminer.pdftypes import PDFObjRef


def guarded_bbox(bbox):
    bbox_guarded = []
    for v in bbox:
        u = v.resolve() if isinstance(v, PDFObjRef) else v
        bbox_guarded.append(u)
    return bbox_guarded
