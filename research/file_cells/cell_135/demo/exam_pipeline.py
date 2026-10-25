"""End-to-end exam flow of the AI University platform:
   booklet → Group 1 digitizer → (Group 12 input guard) → Group 2 grader → marksheet + human-review queue.

    # pages straight from the scanner
    python demo/exam_pipeline.py demo/sample_booklet/T001_p1.png --answer-key demo/answer_key.json \
        --digitizer-url http://127.0.0.1:8000 --grader-url http://127.0.0.1:8100/evaluate-answer \
        --guard-url http://127.0.0.1:8100/check-input --roster demo/roster.txt
    # or a booklet already digitized and corrected in the review UI (GET /ui → "Export reviewed JSON")
    python demo/exam_pipeline.py T001_reviewed.json --answer-key demo/answer_key.json --grader-url ...

Only the digitizer's JSON contract (schema 1.0) is used — no import of our package. An answer is graded automatically
only if the digitizer did not ask for review (or a human approved it) and the guard calls it safe; everything else goes
to the review queue with the reason. Marks are never produced for a booklet without a student ID.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def digitize(pages, url, roster=None, booklet_id=None):
    files = [("pages", (Path(p).name, Path(p).read_bytes(), "application/octet-stream")) for p in pages]
    r = requests.post(f"{url}/digitize", files=files, timeout=600,
                      data={"booklet_id": booklet_id or Path(pages[0]).stem, "roster": ",".join(roster or [])})
    r.raise_for_status()
    return r.json()


def run(booklet: dict, answer_key: dict, grader_url: str, guard_url: str | None = None) -> dict:
    """booklet: the digitizer's JSON (schema 1.0); answer_key: {"Q1": {question_text, reference_answer, rubric, max_marks}}."""
    graded, queue = [], []
    sid = booklet.get("student_id")
    for a in booklet["answers"]:
        item = {"question": a["question"], "page": a["page"], "digitizer_confidence": a["confidence"]}
        approved = (a.get("review") or {}).get("status") in ("approved", "corrected")
        if not sid:
            queue.append({**item, "reason": "student not identified"})
            continue
        if a["needs_review"] and not approved:
            queue.append({**item, "reason": "digitizer: " + (", ".join(a["flags"]) or "low confidence")})
            continue
        if guard_url:
            g = requests.post(guard_url, json={"text": a["answer_text"]}, timeout=60).json()
            if g["risk_level"] != "SAFE":
                queue.append({**item, "reason": f"guard: {g['risk_level']} {g['attack_type']}",
                              "suspicious_text": g["suspicious_text"]})
                continue
        key = answer_key.get(a["question"])
        if key is None:
            queue.append({**item, "reason": "question not in the answer key"})
            continue
        r = requests.post(grader_url, timeout=120, json={
            "question": a["question"], "question_text": key.get("question_text", ""), "reference_answer": key["reference_answer"],
            "rubric": key.get("rubric", []), "student_answer": a["answer_text"], "max_marks": key.get("max_marks", 10)})
        r.raise_for_status()
        g = r.json()
        graded.append({**item, "score": g["score"], "maximum_score": g["maximum_score"], "feedback": g["feedback"],
                       "grader_confidence": g["confidence"], "confidence": round(a["confidence"] * g["confidence"], 3),
                       "human_reviewed": approved})
    missing = sorted(set(answer_key) - {a["question"] for a in booklet["answers"]})
    return {"booklet_id": booklet["booklet_id"], "student_id": sid, "graded": graded, "review_queue": queue,
            "questions_not_found": missing, "auto_marks": sum(g["score"] for g in graded),
            "auto_max": sum(g["maximum_score"] for g in graded)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("inputs", nargs="+", help="page images / PDF of one booklet, or one digitizer JSON")
    ap.add_argument("--answer-key", required=True)
    ap.add_argument("--digitizer-url", default="http://127.0.0.1:8000")
    ap.add_argument("--grader-url", default="http://127.0.0.1:8100/evaluate-answer")
    ap.add_argument("--guard-url", default=None)
    ap.add_argument("--roster", default=None, help="file with one roll number per line")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if len(a.inputs) == 1 and a.inputs[0].endswith(".json"):
        booklet = json.loads(Path(a.inputs[0]).read_text())
    else:
        roster = Path(a.roster).read_text().split() if a.roster else None
        booklet = digitize(a.inputs, a.digitizer_url, roster)
    sheet = run(booklet, json.loads(Path(a.answer_key).read_text()), a.grader_url, a.guard_url)
    text = json.dumps(sheet, indent=1, ensure_ascii=False)
    Path(a.out).write_text(text) if a.out else print(text)
    print(f"{sheet['booklet_id']}: student {sheet['student_id']} · {len(sheet['graded'])} answers graded "
          f"({sheet['auto_marks']:g}/{sheet['auto_max']:g}) · {len(sheet['review_queue'])} to human review"
          + (f" · not found: {', '.join(sheet['questions_not_found'])}" if sheet["questions_not_found"] else ""))