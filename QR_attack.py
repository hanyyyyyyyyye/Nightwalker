import sys
sys.path.append("/home/Newdisk/yanyunjie/code_practics/infraCam/models/yolov5-master")
from models.common import DetectMultiBackend
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import xml.dom.minidom as xmldom
import cv2
from torchvision import transforms
from torch.optim import Adam,SGD
import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unloader=transforms.ToPILImage()
# file_path
model_weight_path="/home/Newdisk/yanyunjie/code_practics/infraCam/models/yolov5-master/runs/train/exp4/weights/last.pt"


def yolo_pre(pre):
    zero=torch.zeros_like(pre[:,4]).to(pre.device)
    return torch.where(pre[:,4]>0.5,pre[:,4],zero).sum()
def load_xml(xml_path):
    xml_flie = xmldom.parse(xml_path)
    eles=xml_flie.documentElement
    return eles

def read_image(image_path):
    from PIL import Image
    from torchvision import datasets, transforms, models
    img = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(img).unsqueeze(0).contiguous()
    return image
def generate_QR_patch(z,tau,patch_size):
    coordinates = torch.stack(torch.meshgrid(torch.ones(z.shape[0]), torch.zeros(z.shape[0])), -1).to(device)
    gb=torch.zeros_like(z).uniform_(0,1)
    gb_log= -(-(gb + 1e-20).log() + 1e-20).log()
    color_map = F.softmax((torch.log(z) + gb_log) / tau, dim=-1)

    patch=coordinates*color_map
    patch=patch[...,0]+patch[...,1]
    # print(patch)
    # patch=((patch>0.5)*1).float().unsqueeze(0)

    scale_factor=patch_size/20

    patch=F.interpolate(patch.unsqueeze(0).unsqueeze(0),scale_factor=scale_factor,mode="nearest")
    ones=torch.ones_like(patch).to(device)
    zeros=torch.zeros_like(patch).to(device)
    patch=torch.where(patch>0.5,ones,zeros).repeat(1,3,1,1)
    patch_img=patch.clone().detach().cpu()
    patch_image=unloader(patch_img.squeeze(0))
    patch_image.save('/home/Newdisk/yanyunjie/code_practics/patch/yolov5-master/result/QR/patch.jpg')
    return patch
def generate_mask(z,xml_path,img):
    eles = load_xml(xml_path)
    mask = torch.zeros_like(img).unsqueeze(0).to(device)
    for i in range(len(eles.getElementsByTagName('name'))):
        if eles.getElementsByTagName('name')[i].firstChild.data == 'person':
            xmin = int(eles.getElementsByTagName('xmin')[i].firstChild.data)
            xmax = int(eles.getElementsByTagName('xmax')[i].firstChild.data)
            ymin = int(eles.getElementsByTagName('ymin')[i].firstChild.data)
            ymax = int(eles.getElementsByTagName('ymax')[i].firstChild.data)
            lx = int((xmin + xmax) / 2 * 0.92)
            ly = int((ymin + ymax) / 2 * 0.89)
            patch_size = int(min((xmax - xmin) *2/ 3, (ymax - ymin)*2 / 3))
            if lx+patch_size>=mask.shape[3] or ly+patch_size>=mask.shape[2]:
                patch_size=int(min(mask.shape[2]-lx,mask.shape[1]-ly))
            adv_patch = generate_QR_patch(z, 0.3,patch_size)

            mask[:,:, ly:ly+patch_size, lx:lx+patch_size] = adv_patch

    return mask

def QR_attack(z,image_path,label_path,e):
    optimizer = Adam([z], lr=0.01)

    model = DetectMultiBackend(model_weight_path, device=device)
    crt = nn.MSELoss()
    for i in tqdm.tqdm(range(e)):
        for name in tqdm.tqdm(os.listdir(image_path)):
            xml_path=os.path.join(label_path, name.split('.')[0] + '.xml')
            image = cv2.imread(os.path.join(image_path, name), 0)
            img = torch.from_numpy(image / 255.0).unsqueeze(0).repeat(3, 1, 1).to(device)
            mask=generate_mask(z,xml_path,img)
            img=torch.where(mask>0,mask,img).float()
            img1 = img.clone().detach().cpu()
            image_result = unloader(img1.squeeze(0))
            image_result.save('/home/Newdisk/yanyunjie/code_practics/infraCam/baseline/QR_attack/yolov5_result/{}'.format(name))
            pre=model(img)[0][0]
            loss1=yolo_pre(pre)
            loss2=1-z.sum()/(z.shape[0]*z.shape[1])
            loss=loss1+loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



if __name__=="__main__":
    image_path = '/home/Newdisk/yanyunjie/code_practics/infraCam/dataset/FLIR_F/exp_Data/images'
    label_path = '/home/Newdisk/yanyunjie/code_practics/infraCam/dataset/FLIR_F/exp_Data/XML'
    z = torch.rand([20, 20, 2]).to(device)
    z = F.softmax(z, dim=-1)
    z.requires_grad=True
    e=100
    QR_attack(z,image_path,label_path,e)
    print("finished...")