import torch
from torch.utils.data import DataLoader, random_split
from dataset import CASIADataset
from model import EmotionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CASIADataset("data/CASIA")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset,[train_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


model = EmotionCNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 40
loss_threshold = 0.05

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for mfcc, label in train_loader:
        mfcc = mfcc.to(device)
        label = label.to(device)

        output = model(mfcc)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    #提前终止机制
    if avg_loss < loss_threshold:
        print("Loss below threshold, early stopping triggered.")
        break

torch.save(model.state_dict(),"emotion_cnn.pth")

print("Model saved to emotion_cnn.pth")


