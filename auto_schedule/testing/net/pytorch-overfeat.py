import torch 
import torch.nn as nn
import time


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
            padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        ret = self.conv(inputs)
        return ret


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inputs):
        return torch.flatten(inputs)


class OverFeat(nn.Module):
    def __init__(self, image_channel=3, num_classes=1470):
        super(OverFeat, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(image_channel, 96, 11, 4, 5),
            nn.MaxPool2d(2, 2),
            ConvBlock(96, 256, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(1024 * 6 * 6, 3072),
            nn.Linear(3072, 4096),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, inputs):
        return self.net(inputs)


if __name__ == "__main__":
    net = OverFeat(3, 1000)
    net.cuda("cuda:0")
    batch_size = 1
    inputs = torch.randn([batch_size, 3, 192, 192]).cuda("cuda:0")
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