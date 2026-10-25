from exam_digitizer.models.layout import crop_line, layout_page
from exam_digitizer.preprocessing import preprocess
from tests.conftest import make_page


def test_lines_and_margin_marks_are_found():
    n_answers, per = 3, 2
    pp = preprocess(make_page(n_answers=n_answers, lines_per_answer=per))
    bands, marks, info = layout_page(pp, 1)
    assert len(bands) == n_answers * per
    assert len(marks) == n_answers
    assert sum(b.header is not None for b in bands) == n_answers
    assert bands == sorted(bands, key=lambda b: b.y0)             # reading order
    crop = crop_line(pp, bands[0])
    assert crop.ndim == 2 and crop.shape[0] > 20 and crop.shape[1] > 200


def test_without_marks_no_headers_are_attached():
    pp = preprocess(make_page(marks=False))
    bands, marks, info = layout_page(pp, 1)
    assert len(bands) == 4 and marks == [] and all(b.header is None for b in bands)
