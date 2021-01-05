from torchvision import transforms
from unet_model import UNet
import numpy as np
import cv2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    model = UNet(3, num_classes).to(device)
    checkpoints = torch.load('model/checkpoint-best.pth')
    model.load_state_dict(checkpoints['state_dict'])

    video = cv2.VideoCapture('test.avi')
    current_frame = 1
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    while True:
        flag, img = video.read()
        current_frame += 1
        if current_frame == total_frame:
            current_frame = 1
            video.set(cv2.CAP_PROP_POS_FRAMES, 1)



if __name__ == '__main__':
    main()
