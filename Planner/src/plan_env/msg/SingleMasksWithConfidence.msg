# SingleMasksWithConfidence.msg
# Per-instance detection result from YOLO segmenter

Header header

# Binary instance mask (mono8: 0=background, 255=object)
sensor_msgs/Image mask

# Human-readable label name (e.g., "car", "person", "chair")
string label_name

# Detection confidence score [0.0, 1.0]
float32 confidence

# Numeric class index from the model (e.g., COCO class id)
int32 label_index
