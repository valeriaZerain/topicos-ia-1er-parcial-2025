import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation
from src.config import get_settings
from src.models import Gun

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.detect_guns(img_array, threshold), img_array

def segment_uploadfile(detector: GunDetector, file, threshold) -> tuple[Segmentation, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.segment_people(img_array, threshold), img_array

@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)

    return results

@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Segmentation:
    segmentation, _ = segment_uploadfile(detector, file, threshold)

    return segmentation

@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    segmentation, img = segment_uploadfile(detector, file, threshold)
    annotated_img = annotate_segmentation(img, segmentation, draw_boxes=draw_boxes)
    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)

    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/detect")
def detect(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
):
    detection, _ = detect_uploadfile(detector, file, threshold)
    file.file.seek(0)
    segmentation, _ = segment_uploadfile(detector, file, threshold)

    return {
        "detection": detection,
        "segmentation": segmentation
    }


    
@app.post("/annotate")
def annotate_all(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    file_bytes = file.file.read()
    file.file.seek(0)

    img_stream = io.BytesIO(file_bytes)
    img_obj = Image.open(img_stream)
    img_array = np.array(img_obj)

    detection = detector.detect_guns(img_array, threshold)
    segmentation = detector.segment_people(img_array, threshold)

    annotated_img = annotate_detection(img_array, detection)
    annotated_img = annotate_segmentation(annotated_img, segmentation)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/guns")
def get_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
):
    detection, _ = detect_uploadfile(detector, file, threshold)

    guns = []
    for i, (label, box) in enumerate(zip(detection.labels, detection.boxes)):
        conf = float(detection.confidences[i]) if i < len(detection.confidences) else None
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        try:
            guns.append(Gun(
                gun_type=label,
                position={"x": cx, "y": cy},
                bbox=box,
                confidence=conf
            ))
        except Exception:
            guns.append({
                "gun_type": label,
                "position": {"x": cx, "y": cy},
                "bbox": box,
                "confidence": conf
            })

    return guns

@app.post("/people")
def get_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
):
    from src.models import Person

    segmentation, _ = segment_uploadfile(detector, file, threshold)

    people = []
    for polygon, label in zip(segmentation.polygons, segmentation.labels):
        poly = np.array(polygon, dtype=np.int32)

        M = cv2.moments(poly)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1]))

        area = float(cv2.contourArea(poly))

        try:
            people.append(
                Person(
                    category=label,
                    position={"x": cx, "y": cy},
                    area=area
                )
            )
        except Exception:
            people.append({
                "category": label,
                "position": {"x": cx, "y": cy},
                "area": area
            })

    return people   

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")
