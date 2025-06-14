import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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
print("\n标签分布:")
print(df['label'].value_counts())


# 2. 文本预处理
def clean_text(text):
    """清理文本数据"""
    text = text.lower()  # 转小写
    text = re.sub(r'[^\w\s]', '', text)  # 移除非字母数字字符
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = re.sub(r'\s+', ' ', text).strip()  # 移除多余空格
    return text


df['cleaned_comment'] = df['comment'].apply(clean_text)


# 3. 构建词汇表
def build_vocab(texts, vocab_size=10000):
    """构建词汇表"""
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)

    # 选择最常见的词
    vocab = {word: idx + 2 for idx, (word, count) in enumerate(word_counts.most_common(vocab_size))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab


vocab = build_vocab(df['cleaned_comment'])
print(f"\n词汇表大小: {len(vocab)}")


# 4. 文本向量化
def text_to_sequence(text, vocab):
    """将文本转换为序列"""
    words = text.split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    return sequence


df['sequence'] = df['cleaned_comment'].apply(lambda x: text_to_sequence(x, vocab))

# 5. 分割数据集
X = df['sequence'].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print(f"\n训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")


# 6. 创建PyTorch数据集
class CommentDataset(Dataset):
    """自定义PyTorch数据集类"""

    def __init__(self, sequences, labels, max_length=100):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # 截断或填充序列
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [0] * (self.max_length - len(sequence))  # 用0填充

        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float)


# 创建数据集和数据加载器
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
    """情感分类LSTM模型"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        # 连接最后两个LSTM层的隐藏状态
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)


# 模型参数
vocab_size = len(vocab)
embedding_dim = 128
hidden_dim = 128
output_dim = 1
n_layers = 2
dropout = 0.5

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)


# 8. 训练模型
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    """训练模型并返回训练历史"""
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(sequences).squeeze(1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted_labels = torch.round(torch.sigmoid(predictions))
            train_correct += (predicted_labels == labels).sum().item()
            train_total += labels.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)

                predictions = model(sequences).squeeze(1)
                loss = criterion(predictions, labels)

                val_loss += loss.item()
                predicted_labels = torch.round(torch.sigmoid(predictions))
                val_correct += (predicted_labels == labels).sum().item()
                val_total += labels.size(0)

        # 计算指标
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    return history


# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失

# 训练模型
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
def evaluate_model(model, test_loader):
    """评估模型在测试集上的表现"""
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            predictions = model(sequences).squeeze(1)
            loss = criterion(predictions, labels)

            test_loss += loss.item()
            predicted_labels = torch.round(torch.sigmoid(predictions))
            test_correct += (predicted_labels == labels).sum().item()
            test_total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted_labels.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    test_acc = test_correct / test_total

    print(f"\n测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['负向', '正向']))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负向', '正向'],
                yticklabels=['负向', '正向'])
    plt.ylabel('实际标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.show()

    return test_acc


print("\n评估模型在测试集上的表现...")
test_acc = evaluate_model(model, test_loader)

# 11. 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'max_length': max_length
}, 'sentiment_lstm_model.pth')
print("\n模型已保存为 'sentiment_lstm_model.pth'")


# 12. 加载模型进行预测
class SentimentPredictor:
    """情感预测器"""

    def __init__(self, model_path, device='cpu'):
        checkpoint = torch.load(model_path, map_location=device)
        self.vocab = checkpoint['vocab']
        self.max_length = checkpoint['max_length']

        # 初始化模型
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
        """预测文本的情感倾向"""
        # 清理文本
        cleaned_text = clean_text(text)
        # 转换为序列
        sequence = text_to_sequence(cleaned_text, self.vocab)
        # 截断或填充序列
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + [0] * (self.max_length - len(sequence))

        # 转换为tensor
        tensor_sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            prediction = self.model(tensor_sequence).squeeze(1)
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