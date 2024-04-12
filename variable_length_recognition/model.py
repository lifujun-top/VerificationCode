import torch
from torch import nn
from torchvision.models import resnet18


class LstmCtcNet(nn.Module):

    def __init__(self, image_shape, label_map_length):
        super(LstmCtcNet, self).__init__()
        self.bone = nn.Sequential(*list(resnet18().children())[:-3])
        # 计算shape
        x = torch.zeros((1, 3) + image_shape)
        shape = self.bone(x).shape  # [1, 256, 4, 10] BATCH, DIM, HEIGHT, WIDTH
        bone_output_shape = shape[1] * shape[2]

        self.lstm = nn.LSTM(bone_output_shape, bone_output_shape, num_layers=1, bidirectional=True)
        self.embedding = nn.Linear(bone_output_shape * 2, label_map_length)

    def forward(self, x):
        x = self.bone(x)
        x = x.permute(3, 0, 1, 2)  # [10, 1, 256, 4]
        w, b, c, h = x.shape
        x = x.view(w, b, c * h)    # [10, 1, 256 * 4] time_step batch_size input

        x, _ = self.lstm(x)
        time_step, batch_size, hidden = x.shape     # [10, 1, 2048]  time_step batch_size hidden
        x = x.view(time_step * batch_size, hidden)
        x = self.embedding(x)    # [time_step * batch_size, label_map_length]
        return x.view(time_step, batch_size, -1)  # [time_step, batch_size, label_map_length] [10, 1, 36]

