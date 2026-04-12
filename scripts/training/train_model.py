import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from models import CRNN

class HanNomDataset(Dataset):
    def __init__(self, data_dir, labels_file=None, transform=None, vocab=None):
        self.data_dir = data_dir
        self.transform = transform
        self.vocab = vocab
        self.char2idx = {char: i + 1 for i, char in enumerate(vocab)}
        self.idx2char = {i + 1: char for i, char in enumerate(vocab)}
        
        if labels_file:
            # Real-world data (NomNaOCR format)
            self.samples = []
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        self.samples.append((parts[0], parts[1]))
        else:
            # Synthetic data (organized by char_hex folders)
            self.samples = []
            for char_hex in os.listdir(data_dir):
                char_path = os.path.join(data_dir, char_hex)
                if os.path.isdir(char_path):
                    char = chr(int(char_hex, 16))
                    for img_name in os.listdir(char_path):
                        self.samples.append((os.path.join(char_hex, img_name), char))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        # Convert label to indices
        label_indices = [self.char2idx.get(c, 0) for c in label]
        label_len = len(label_indices)
        
        return image, torch.IntTensor(label_indices), torch.IntTensor([label_len])

def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, labels, label_lengths) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images) # [T, batch, num_classes]
            
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.int32)
            
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {total_loss / len(train_loader):.4f}")

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_h = 64
    batch_size = 32
    
    # Load vocab from CSVs
    df1 = pd.read_csv('data/Unihan_Vietnamese.csv')
    df2 = pd.read_csv('data/Thieu_Chuu_Dictionary.csv')
    vocab = sorted(list(set(df1['char'].dropna()) | set(df2['char'].dropna())))
    num_classes = len(vocab) + 1 # +1 for blank token
    
    model = CRNN(num_classes=num_classes, img_h=img_h).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    transform = transforms.Compose([
        transforms.Resize((img_h, img_h)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Stage 1: Synthetic Data
    synthetic_dir = "dataset"
    if os.path.exists(synthetic_dir) and os.listdir(synthetic_dir):
        print("Starting Stage 1: Pre-training on Synthetic Data...")
        train_dataset = HanNomDataset(synthetic_dir, transform=transform, vocab=vocab)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train(model, train_loader, criterion, optimizer, device, epochs=5)
        torch.save(model.state_dict(), 'checkpoints/stage1_model.pth')
    else:
        print("Synthetic dataset not found. Skipping Stage 1.")

    # Stage 2: Real-world Data (Fine-tuning)
    real_dir = "data/NomNaOCR_Processed/images"
    labels_file = "data/NomNaOCR_Processed/labels.txt"
    if os.path.exists(real_dir) and os.path.exists(labels_file):
        print("Starting Stage 2: Fine-tuning on Real-world Data...")
        # Load Stage 1 weights if available
        if os.path.exists('checkpoints/stage1_model.pth'):
            model.load_state_dict(torch.load('checkpoints/stage1_model.pth'))
        
        train_dataset = HanNomDataset(real_dir, labels_file=labels_file, transform=transform, vocab=vocab)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train(model, train_loader, criterion, optimizer, device, epochs=10)
        torch.save(model.state_dict(), 'checkpoints/final_model.pth')
    else:
        print("Real-world dataset not found. Skipping Stage 2.")

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    main()
