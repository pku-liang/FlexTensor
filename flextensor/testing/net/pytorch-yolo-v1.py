import torch 
import torch.nn as nn
import time


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = nn.ReLU()

    def forward(self, inputs):
        ret = self.conv(inputs)
        ret = self.act(ret)
        return ret


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inputs):
        return torch.flatten(inputs)


class YOLO(nn.Module):
    def __init__(self, image_channel=3, num_classes=1470):
        super(YOLO, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(image_channel, 64, 7, 2, 3),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(192, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 2, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
            Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, inputs):
        return self.net(inputs)


if __name__ == "__main__":
    net = YOLO(3, 1470)
    net.cuda("cuda:0")
    batch_size = 1
    inputs = torch.randn([batch_size, 3, 448, 448]).cuda("cuda:0")
    output = net(inputs)

    torch.cuda.synchronize()
    beg = time.time()
    device_time = 0.0
    for i in range(50):
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        start.record()
        net(inputs)
        finish.record()
        torch.cuda.synchronize()
        device_time += start.elapsed_time(finish)
    end = time.time()
    print("Host time pass {}ms".format((end - beg) * 1e3 / 50))
    print("Device time pass {}ms".format(device_time / 50))