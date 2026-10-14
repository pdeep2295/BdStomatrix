from ultralytics import YOLO

LAST_CHECKPOINT = (
    "/Users/pawandeepsingh/Documents/Postdoc/YOLO_model/"
    "Training_Results/yolo26n_Stomata_DIC_Final/weights/last.pt"
)

model = YOLO(LAST_CHECKPOINT)
results = model.train(resume=True)