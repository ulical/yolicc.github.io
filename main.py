import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong', 'Arial Unicode MS']  # 任选系统中有的
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保可重现性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 加载和探索数据
df = pd.read_csv('comments.csv')
print("\n数据集概览:")
print(df.head())
# 兼容不同csv格式，自动查找评论列
for col in df.columns:
    if '评论' in col or 'review' in col or 'comment' in col:
        text_col = col
        break
label_col = [col for col in df.columns if 'label' in col or '标签' in col][0]

# 只保留有用的列
df = df[[text_col, label_col]].rename(columns={text_col: 'comment', label_col: 'label'})

# 修正标签类型
df['label'] = df['label'].astype(int)

print("\n标签分布:")
print(df['label'].value_counts())

# 2. 文本预处理
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_comment'] = df['comment'].apply(clean_text)

# 3. 构建词汇表
def build_vocab(texts, vocab_size=10000):
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.most_common(vocab_size))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(df['cleaned_comment'])
print(f"\n词汇表大小: {len(vocab)}")

# 4. 文本向量化
def text_to_sequence(text, vocab):
    words = text.split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    return sequence

df['sequence'] = df['cleaned_comment'].apply(lambda x: text_to_sequence(x, vocab))

# 5. 分割数据集
X = df['sequence'].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

print(f"\n训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")

# 6. 创建PyTorch数据集
class CommentDataset(Dataset):
    def __init__(self, sequences, labels, max_length=100):
        self.sequences = [seq for seq in sequences]
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float)

max_length = 100
batch_size = 64

train_dataset = CommentDataset(X_train, y_train, max_length)
val_dataset = CommentDataset(X_val, y_val, max_length)
test_dataset = CommentDataset(X_test, y_test, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 7. 定义LSTM模型
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        # 拼接最后一层的正向和反向
        hidden_cat = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden_cat).squeeze(1)

vocab_size = len(vocab)
embedding_dim = 128
hidden_dim = 128
output_dim = 1
n_layers = 2
dropout = 0.5

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)

# 8. 训练模型
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.round(torch.sigmoid(predictions))
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                predictions = model(sequences)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                preds = torch.round(torch.sigmoid(predictions))
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    return history

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print("\n开始训练模型...")
history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=15)

# 9. 可视化训练过程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# 10. 评估模型
def evaluate_model(model, data_loader, set_name="测试集"):
    model.eval()
    loss_total = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            loss_total += loss.item()
            preds = torch.round(torch.sigmoid(predictions))
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    avg_loss = loss_total / len(data_loader)
    accuracy = correct / total
    print(f"\n{set_name}损失: {avg_loss:.4f}")
    print(f"{set_name}准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['负向', '正向']))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负向', '正向'], yticklabels=['负向', '正向'])
    plt.ylabel('实际标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    plt.savefig(f'{set_name}_confusion_matrix.png')
    plt.show()
    return accuracy

print("\n评估模型在验证集上的表现...")
val_acc = evaluate_model(model, val_loader, set_name="验证集")
print("\n评估模型在测试集上的表现...")
test_acc = evaluate_model(model, test_loader, set_name="测试集")

# 11. 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'max_length': max_length,
    'embedding_dim': embedding_dim,
    'hidden_dim': hidden_dim,
    'output_dim': output_dim,
    'n_layers': n_layers,
    'dropout': dropout
}, 'sentiment_lstm_model.pth')
print("\n模型已保存为 'sentiment_lstm_model.pth'")

# 12. 加载模型进行预测
class SentimentPredictor:
    def __init__(self, model_path, device='cpu'):
        checkpoint = torch.load(model_path, map_location=device)
        self.vocab = checkpoint['vocab']
        self.max_length = checkpoint['max_length']
        embedding_dim = checkpoint.get('embedding_dim', 128)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        output_dim = checkpoint.get('output_dim', 1)
        n_layers = checkpoint.get('n_layers', 2)
        dropout = checkpoint.get('dropout', 0.5)
        model = SentimentLSTM(
            vocab_size=len(self.vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model
        self.device = device

    def predict_sentiment(self, text):
        cleaned_text = clean_text(text)
        sequence = text_to_sequence(cleaned_text, self.vocab)
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        tensor_sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(tensor_sequence)
            probability = torch.sigmoid(prediction).item()
        sentiment = "正向" if probability > 0.5 else "负向"
        confidence = probability if sentiment == "正向" else 1 - probability
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'probability': probability
        }

# 13. 示例预测
print("\n加载模型进行预测...")
predictor = SentimentPredictor('sentiment_lstm_model.pth', device=device)

test_samples = [
    "这苹果又甜又脆，非常新鲜！",
    "收到的香蕉都烂了，质量太差",
    "水果品质一般，价格偏高",
    "包装精美，送货速度快，水果新鲜",
    "部分水果有瑕疵，但整体还算满意"
]

print("\n示例预测:")
for sample in test_samples:
    result = predictor.predict_sentiment(sample)
    print(f"文本: '{result['text']}'")
    print(f"情感: {result['sentiment']} (置信度: {result['confidence']:.4f})")
    print(f"正向概率: {result['probability']:.4f}")
    print("-" * 50)