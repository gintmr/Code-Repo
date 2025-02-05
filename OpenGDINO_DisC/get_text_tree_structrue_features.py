
from models.GroundingDINO import TextFeatureExtractor
from tqdm import tqdm
noun_list = categories = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# 使用 tqdm 包装 noun_list
for noun in tqdm(noun_list, desc="Processing nouns"):
    try:
        # 初始化 TextFeatureExtractor
        Extractor = TextFeatureExtractor(noun)
        Extractor.to("cpu")
        result = Extractor.forward()
        print(f"Processed: {noun}")
    except Exception as e:
        print(f"Error processing {noun}: {e}")

print("All done!!!")