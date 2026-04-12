import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, img_h=64, nc=1, n_hidden=256):
        super(CRNN, self).__init__()
        
        # CNN layers to extract features
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 64x32x32
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 128x16x16
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # 256x8x17
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # 512x4x18
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True) # 512x3x17
        )
        
        # RNN layers (Bi-LSTM)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, n_hidden, n_hidden),
            BidirectionalLSTM(n_hidden, n_hidden, num_classes)
        )

    def forward(self, x):
        # x: [batch, 1, 64, 64]
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "The height of conv must be 1"
        conv = conv.squeeze(2) # [batch, 512, width]
        conv = conv.permute(2, 0, 1) # [width, batch, 512]
        
        output = self.rnn(conv)
        return output

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec) # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

if __name__ == "__main__":
    # Test the model
    model = CRNN(num_classes=1000)
    input = torch.randn(1, 1, 64, 64)
    output = model(input)
    print(f"Output shape: {output.shape}") # Expected: [T, batch, num_classes]
