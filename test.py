import torch
from model import EmotionCNN
from utils import extract_mfcc

emotion_map = {
    0: "angry",
    1: "fear",
    2: "happy",
    3: "neutral",
    4: "sad",
    5: "surprise"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN()
model.load_state_dict(torch.load("emotion_cnn.pth"))
model.to(device)
model.eval()

wav_path = "test.wav"

mfcc = extract_mfcc(wav_path)
mfcc = torch.tensor(mfcc).float().transpose(0, 1).unsqueeze(0).unsqueeze(0)
mfcc = mfcc.to(device)

with torch.no_grad():
    output = model(mfcc)
    pred = torch.argmax(output, dim=1).item()

print("Predicted emotion:", emotion_map[pred])
