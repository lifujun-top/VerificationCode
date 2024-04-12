import random
import string
from tqdm import tqdm
from captcha.image import ImageCaptcha


SAMPLE_NUMBER = 40000                                       # 生成个数
CHAR_SET = [c for c in 'abcdefhjkmnpqrstuvwxyz234568']      # 字符集

image = ImageCaptcha()
for _ in tqdm(range(SAMPLE_NUMBER)):
    answer = ''.join(random.choices(CHAR_SET, k=random.randint(4, 6)))
    name = ''.join(random.choices(string.ascii_lowercase, k=16))
    # 用下划线分割, 加入随机文本, 防止重复
    folder = "valid" if random.random() > 0.98 else "train"
    image.write(answer, 'samples/%s/%s_%s.png' % (folder, answer, name))
