import cv2
import numpy as np
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def get_image(self, pts):
        return self.decoder(pts)

def get_vector(x, y, img_size=280, scale=1):
	x -= img_size/2
	y -= img_size/2
	x *= scale
	y *= scale
	return [x,y]

def drawCircle(event, x, y, flags, param):
	print(f'{x}, {y}')
	pt = get_vector(x, y)
	pt = torch.Tensor(pt)
	newImg = model.get_image(pt).detach().numpy().reshape(28,28)
	newImg = cv2.resize(newImg, (280,280))
	cv2.circle(newImg, (x,y), 10, (255,0,0), -1)
	cv2.imshow('image', newImg)


if __name__ == '__main__':
	model = AutoEncoder()
	model.load_state_dict(torch.load('model.pth'))
	img = cv2.imread('start.jpg')
	cv2.imshow('image', img)

	cv2.setMouseCallback('image', drawCircle)

	cv2.waitKey(0)
	cv2.destroyAllWindows()