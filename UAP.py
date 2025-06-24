import os
import sys
sys.path.append("/home/Newdisk/yanyunjie/code_practics/infraCam/models/yolov5-master")
from models.common import DetectMultiBackend
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trans = transforms.Compose([
                transforms.ToTensor(),
            ])
def UAP():
    weights_path = "/home/Newdisk/yanyunjie/code_practics/infraCam/models/yolov5-master/runs/train/exp4/weights/best.pt"
    model = DetectMultiBackend(weights_path, device=device)
    model.eval()
    img_root="/home/Newdisk/yanyunjie/code_practics/infraCam/dataset/FLIR_F/exp_Data/images"
    for img_name in os.listdir(img_root):
        img_path=os.path.join(img_root,img_name)
        infrared_input=Image.open(img_path)
        infrared_det=trans(infrared_input).unsqueeze(0)
        infrared_det = F.interpolate(infrared_det, (640, 640), mode='bilinear', align_corners=False)



