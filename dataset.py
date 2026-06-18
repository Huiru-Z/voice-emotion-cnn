import os
import torch
from torch.utils.data import Dataset
from utils import extract_mfcc

emotion_dict = {
    "angry": 0,
    "fear": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "surprise": 5
}


class CASIADataset(Dataset):

    def __init__(self, root_dir):
        self.samples = []

        for emotion in emotion_dict.keys():
            emotion_dir = os.path.join(root_dir, emotion)
            for file in os.listdir(emotion_dir):
                if file.endswith(".wav"):
                    self.samples.append((os.path.join(emotion_dir, file),emotion_dict[emotion]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]

        mfcc = extract_mfcc(wav_path)

        # (T, F) → (1, F, T)
        mfcc = torch.tensor(mfcc).float().transpose(0, 1).unsqueeze(0)

        label = torch.tensor(label).long()

        return mfcc, label
