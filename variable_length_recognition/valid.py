import torch
from tqdm import tqdm
from model import LstmCtcNet
from torch.utils.data import DataLoader
from MyImageDataset import ImageDataset
from MyUtils import tensor_to_str, ctc_to_str
import torchvision.transforms as T


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CAPTCHA_MAX_LENGTH = 5                        # 验证码长度
IMAGE_SHAPE = (64, 160)

valid_set = ImageDataset(
    './samples/valid',
    maxLength=CAPTCHA_MAX_LENGTH,
    transform=T.Compose([
        T.ToPILImage(),
        T.Resize(IMAGE_SHAPE),
        T.ToTensor(),
        T.Normalize((0.79490087, 0.79427771, 0.79475806), (0.30808181, 0.30900241, 0.30821851))
    ])
)

CHAR_SET = valid_set.get_label_map()
CAPTCHA_CHARS = len(CHAR_SET)

valid_loader = DataLoader(dataset=valid_set, batch_size=2)
model = LstmCtcNet(IMAGE_SHAPE, CAPTCHA_CHARS)
model.load_state_dict(torch.load("models/save_20.model"))
model = model.to(DEVICE)


if __name__ == '__main__':
    model.eval()
    correct = count = 0
    for images, texts, target_lengths in tqdm(valid_loader):
        images = images.to(DEVICE)

        predicts = model(images)
        for i in range(predicts.shape[1]):
            predict = predicts[:, i, :]
            predict = predict.argmax(1)
            predict = predict.contiguous()
            count += 1
            label_text = tensor_to_str(texts[i], CHAR_SET)[:target_lengths[i]]
            predict_text = ctc_to_str(predict, CHAR_SET)
            predict_text_md = tensor_to_str(predict, CHAR_SET)
            print(label_text, '->', predict_text, '(', predict_text_md, ')')
            if label_text == predict_text:
                correct += 1

    print('Acc', correct / count)
