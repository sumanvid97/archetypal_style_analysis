import os
import torch
from log_utils import get_logger
from im_utils import load_img
from torch.utils.data import Dataset

log = get_logger()
supported_img_formats = ('.png', '.jpg', '.jpeg')

class PaintingsDataset(Dataset):

    def __init__(self, args):
        super(Dataset, self).__init__()

        self.contentSize = args.contentSize
        self.dataDir = args.dataDir
        self.paintings = []
        for root, dirs, files in os.walk(args.dataDir):
            for file in files:
                path = os.path.join(root,file)
                self.paintings.append(path)
        # print(self.paintings)
        log.info('Added ' + str(len(self.paintings)) + ' paintings to the dataset from ' + str(self.dataDir))        

    def __len__(self):
        return len(self.paintings)

    def __getitem__(self, idx):
        name = self.paintings[idx]
        path = os.path.join(self.dataDir, name)
        content = load_img(path, self.contentSize)
        return {'content': content, 'path': path, 'name': name}
