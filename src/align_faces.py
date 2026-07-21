from facenet_pytorch import MTCNN
from PIL import Image

# keep_all=False -> if MTCNN finds multiple faces, only keeps most confident one
mtcnn = MTCNN(keep_all=False, device='cpu')

# Load test image w/ PIL
img_path = "data/lfw_subset/George_W_Bush/George_W_Bush_0001.jpg"
img = Image.open(img_path)

print(mtcnn.detect(img))