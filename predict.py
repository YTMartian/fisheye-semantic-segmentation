from torchvision import transforms
from unet_model import UNet
import numpy as np
import torch
import sys
import cv2

colors = {0: [0, 0, 0], 1: [255, 179, 120], 2: [224, 204, 151], 3: [236, 121, 154], 4: [159, 2, 81], 5: [206, 235, 251],
          6: [102, 167, 197], 7: [33, 182, 168], 8: [127, 23, 31], 9: [182, 119, 33], 10: [127, 84, 23]}

def label_to_image(label):
    shape = label.shape
    result = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[i, j] = colors[label[i, j]]
    # result = result[:, :, (2, 1, 0)]  # BGR to RGB.
    return result

def main(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    model = UNet(3, num_classes).to(device)
    checkpoints = torch.load('model/checkpoint-best.pth')
    model.load_state_dict(checkpoints['state_dict'])
    model.eval()

    video = cv2.VideoCapture(video_path)
    current_frame = 1
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    while True:
        flag, image = video.read()
        current_frame += 1
        if current_frame == total_frame:
            current_frame = 1
            video.set(cv2.CAP_PROP_POS_FRAMES, 1)
        img = transform(image) #(3,512,512)
        img = img.unsqueeze_(0)#(1,3,512,512)
        img = img.to(device, dtype=torch.float)
        pred = model(img)#(1,10,512,512)
        pred = torch.argmax(pred, 1, keepdim=True)#(1,1,512,512)
        pred = pred.squeeze_(0)#(1,512,512)
        pred = pred.squeeze_(0)#(512,512)
        result = label_to_image(pred.cpu().numpy())
        result = np.hstack((image,result))
        cv2.imshow('',result)
        cv2.waitKey(1)



if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'test.avi')
