import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from pygame.locals import *

pygame.init()

# Screen dimensions
boxSize=364
width, height = boxSize,boxSize

# Colors
white = (255, 255, 255)
black = (0, 0, 0)

# Line thickness
thickness = 13

# Set up the screen
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Draw with Thickness")

# Fill the screen with white
screen.fill(white)

# Start the main loop
drawing = False
active=True
lst=[]


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 32)
        self.l4 = nn.Linear(32, 32)
        self.l5 = nn.Linear(32, 32)
        self.l6 = nn.Linear(32, 32)
        self.l7 = nn.Linear(32, 32)
        self.l8 = nn.Linear(32, 10)

    def forward(self, X):
        X = F.relu(self.l1(X))
        X = F.relu(self.l2(X))
        X = F.relu(self.l3(X))
        X = F.relu(self.l4(X))
        X = F.relu(self.l5(X))
        X = F.relu(self.l6(X))
        X = F.relu(self.l7(X))
        X = self.l8(X)
        return F.log_softmax(X, 1)
model = net()
model.load_state_dict(torch.load('ms1.pth',weights_only=True))
model.eval()

def to_tensor(show=False):
    ht=int(boxSize/28)
    l=torch.zeros(784)
    for i in range(28):
        for j in range(28):
            m=0
            for ik in range(ht):
                for jk in range(ht):
                    color = screen.get_at((i*ht+ik,j*ht+jk))
                    m+=255-(color[0]+color[1]+color[2])/3
            l[j*28+i]=m
    max_val = torch.max(l)
    for i in range(784):
        l[i]=int((l[i]/max_val)*256)
    if show:
        plt.imshow(l.view(28,28))
        plt.show()
    output = torch.argmax(model(l.view(-1,784)))

    print("Model output:", output)

show=False
while active:
    # Update the display
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            active = False
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == K_s:
                show=True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Start drawing when mouse button is pressed
            if event.button == 1:  # Left mouse button
                drawing = True
            if event.button == 3: # right?
                to_tensor(show)
                screen.fill(white)
        elif event.type == pygame.MOUSEBUTTONUP:
            # Stop drawing when mouse button is released
            if event.button == 1:
                drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            # Draw when the mouse is moved while the button is pressed
            pygame.draw.circle(screen, black, event.pos, thickness)