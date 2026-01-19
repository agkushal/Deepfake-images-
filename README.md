# Deepfake-images
Deepfake Image Detection (DeepFakeCNNv2)

This project implements a lightweight CNN-based deepfake image detection system that classifies facial images as Real or Fake by learning texture-level and pixel-level inconsistencies introduced during deepfake generation.

Model
- Custom CNN architecture (DeepFakeCNNv2)
- Input size: 128×128 RGB images
- Uses Conv–BatchNorm–ReLU blocks, MaxPooling, AdaptiveAvgPooling, and Dropout
- Binary classification output (Real / Fake)

Dataset
- Approximately 190,000 facial images (Real and Fake)
- Sources include FaceForensics++, Celeb-DF, and DFDC
- Folder structure:
  train/real, train/fake
  test/real, test/fake
- Dataset not included due to size constraints

Results
- Accuracy: 90.58%
- Precision: 0.9121
- Recall: 0.9034
- F1-score: 0.9077

Run
train.py

Conclusion
The project demonstrates that a well-designed CNN can achieve strong deepfake detection performance with low computational cost, making it suitable for real-time and resource-constrained applications.
