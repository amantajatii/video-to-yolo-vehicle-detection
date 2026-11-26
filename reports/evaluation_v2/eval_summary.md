# Detector evaluation

Split `test` · 54 images · 960 px · mps

| model | kind | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AP_mobil | AP50_mobil | AP_truk | AP50_truk | AP_bus | AP50_bus | GT_small | GT_medium | GT_large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best | fine-tuned | 0.5404 | 0.7782 | 0.5462 | 0.2931 | 0.5413 | 0.7043 | 0.6933 | 0.9198 | 0.4293 | 0.6399 | 0.4988 | 0.7750 | 414 | 410 | 122 |

COCO AP uses confidence 0.001, a cap of 300 detections per image, and original-image box areas. Small is below 32² px², medium is 32² to below 96² px², and large is 96² px² or above. — marks unavailable metrics.

Fine-tuned mean inference time was 7.32 ms/image with one image per call, excluding preprocessing, postprocessing and warm-up.

## Error analysis

Confidence ≥ 0.25, IoU ≥ 0.5. Matched wrong-class predictions count as one FP for the predicted class and one FN for the GT class. The size table counts unmatched GT only for FN. In errors.csv, class confusions use the GT box.

| Count | small | medium | large |
| --- | --- | --- | --- |
| GT | 414 | 410 | 122 |
| FN | 63 | 27 | 3 |

| Class | precision | recall | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| mobil | 0.7914 | 0.8991 | 793 | 209 | 89 |
| truk | 0.4714 | 0.6111 | 33 | 37 | 21 |
| bus | 0.5000 | 0.7000 | 7 | 7 | 3 |
