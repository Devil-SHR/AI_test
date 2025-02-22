from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 로컬 이미지 경로 (자신의 경로로 수정)
image_path = "/Users/robor/Desktop/Test/image/다운로드(1).jpeg"
image = Image.open(image_path)

# 모델과 프로세서 로드
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# 이미지 전처리 및 모델 예측
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# 바운딩 박스와 클래스 라벨을 COCO API 형식으로 변환
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# 결과를 화면에 표시
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

# 각 객체에 대해 바운딩 박스를 그리고 라벨과 신뢰도 표시
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    
    # 바운딩 박스 그리기
    rect = patches.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    
    # 라벨과 신뢰도 텍스트 추가
    ax.text(
        box[0], box[1], f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}",
        bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black'
    )

# 결과 이미지 보여주기
plt.show()