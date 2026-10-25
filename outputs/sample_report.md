# Sample application run

Input: `demo/sample_booklet/T001_p1.png`

Student: **23D019** (matches the supplied ground truth). Pages: 1. Answers: 4.

OCR pipeline runtime: 7.106 seconds; model loading, NER and figure detection are additional.

| Question | Page | Confidence | Human review requested |
|---|---:|---:|---|
| Q4 | 1 | 95.0% | No |
| Q2 | 1 | 84.8% | No |
| Q1 | 1 | 95.6% | No |
| Q3 | 1 | 87.9% | No |

## Transcriptions

### Q4

let us consider first what would appear to be the simplest case - that of flat fish on the bottom . The evidence suggest that the fish are

### Q2

aware of the moving footrope before it' reaches them . They move away before

### Q1

it , swimming at the same speed for an unknown period of time , until

### Q3

they become ( 1) tired , or ( 2 ) more used to the disturbance .

## Named entities

| Text | Label | Answer |
|---|---|---|
| first | ORDINAL | Q4 |
| 1 | CARDINAL | Q3 |
| 2 | CARDINAL | Q3 |

Figure candidates detected: 0.

## Comparison with the supplied sample ground truth

Character error rate: 0.86%. Word error rate: 2.70%. Question segmentation F1: 1.000.

These measurements describe this single bundled sample, not overall production accuracy. The transcriptions above are the actual model output and have not been manually corrected.

Full JSON with line boxes: [sample_analysis.json](sample_analysis.json). Original digitizer contract: [sample_digitization.json](sample_digitization.json).
