# Detector evaluation

Split `test` · 54 images · 960 px · mps

| model | kind | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AP_mobil | AP50_mobil | AP_truk | AP50_truk | AP_bus | AP50_bus | GT_small | GT_medium | GT_large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best | fine-tuned | 0.6122 | 0.8415 | 0.7009 | 0.3055 | 0.6414 | 0.7688 | 0.7095 | 0.9313 | 0.5661 | 0.7669 | 0.5611 | 0.8262 | 414 | 410 | 122 |
| yolo11n | COCO | 0.4709 | 0.6221 | 0.4773 | 0.1804 | 0.4595 | 0.6418 | 0.5731 | 0.7894 | 0.3344 | 0.3964 | 0.5051 | 0.6805 | 414 | 410 | 122 |
| yolo11x | COCO | 0.7358 | 0.8333 | 0.8009 | 0.5213 | 0.6580 | 0.8241 | 0.7794 | 0.9130 | 0.5362 | 0.6033 | 0.8919 | 0.9835 | 414 | 410 | 122 |

COCO AP uses confidence 0.001, a cap of 300 detections per image, and original-image box areas. Small is below 32² px², medium is 32² to below 96² px², and large is 96² px² or above. — marks unavailable metrics.

Fine-tuned mean inference time was 7.02 ms/image with one image per call, excluding preprocessing, postprocessing and warm-up.

## Error analysis

Confidence ≥ 0.25, IoU ≥ 0.5. Matched wrong-class predictions count as one FP for the predicted class and one FN for the GT class. The size table counts unmatched GT only for FN. In errors.csv, class confusions use the GT box.

| Count | small | medium | large |
| --- | --- | --- | --- |
| GT | 414 | 410 | 122 |
| FN | 52 | 19 | 3 |

| Class | precision | recall | TP | FP | FN |
| --- | --- | --- | --- | --- | --- |
| mobil | 0.7808 | 0.9206 | 812 | 228 | 70 |
| truk | 0.6531 | 0.5926 | 32 | 17 | 22 |
| bus | 0.4667 | 0.7000 | 7 | 8 | 3 |
