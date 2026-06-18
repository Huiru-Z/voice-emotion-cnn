# utils.py
import librosa
import numpy as np

def extract_mfcc(
        wav_path,
        sr=16000,
        n_mfcc=40,
        max_len=300
    ):
    """
    提取 MFCC 特征并进行长度对齐
    """

    # 加载语音
    signal, sr = librosa.load(wav_path, sr=sr)

    # 提取 MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc
    )

    # 转置为 (time, mfcc)
    mfcc = mfcc.T

    # 长度统一
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:max_len]

    return mfcc
