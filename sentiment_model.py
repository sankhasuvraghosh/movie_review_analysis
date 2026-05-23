import torch
import torch.nn as nn
import pandas as pd 
from torch.utils.data import Dataset,DataLoader ,random_split
import torch.optim as opt
from collections import Counter
import re
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def remove_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

df=pd.read_csv("IMDB Dataset.csv")
df['review']=df['review'].apply(remove_tags)
df['sentiment']=df['sentiment'].apply(lambda x:1 if x== 'positive' else 0)
def build_vocab(texts,max_vocab=30000):
  counter=Counter()
  for text in  texts:
    counter.update(tokenize(text)) 
  vocab={'<PAD>':0,'<UNK>':1}
  for word,_ in counter.most_common(max_vocab):
    vocab[word]=len(vocab)
  return vocab
vocab=build_vocab(df['review'])
print("Vocab size : ",len(vocab))
class text_dataset(Dataset):
  def __init__(self,texts,review,vocab,max_len=300):
    self.data=[]
    self.labels=review
    for text in texts:
      ids    = [vocab.get(t, 1) for t in tokenize(text)]
      ids=ids[:max_len] +[0]*(max_len-len(ids))
      self.data.append(ids)
  def __len__(self):
    return len(self.data)
  def __getitem__(self,idx):
    return (torch.tensor(self.data[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float))
max_len=300
dataset=text_dataset(df['review'].to_list(),df['sentiment'].to_list(),vocab,max_len)
dataloader=DataLoader(dataset,batch_size=128,shuffle=True)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc      = nn.Linear(fc_input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.sigmoid(self.fc(self.dropout(hidden))).squeeze(1)


BATCH_SIZE    = 128
EMBEDDING_DIM = 64
HIDDEN_DIM    = 128
N_LAYERS      = 2
BIDIRECTIONAL = True
DROPOUT       = 0.3
LR            = 0.001
EPOCHS        = 10
MAX_LEN=300
CHECKPOINT    = "sentiment_model.pt"
n  = len(dataset)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
n_test  = n - n_train - n_val

train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE) 

print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}\n")
# --- Model ---
model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
    ).to(device)

optimizer = opt.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

    # --- Training loop ---
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss  = criterion(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss    += loss.item()
        train_correct += (torch.round(preds) == batch_y).sum().item()
        train_total   += batch_y.size(0)
    train_loss /= len(train_loader)
    train_acc   = train_correct / train_total
 
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds     = model(batch_x)
            val_loss += criterion(preds, batch_y).item()
            val_correct += (torch.round(preds) == batch_y).sum().item()
            val_total   += batch_y.size(0)
    val_loss /= len(val_loader)
    val_acc   = val_correct / val_total
 
    print(      f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )
 
        # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':          val_acc,
                'vocab':            vocab,
                'hyperparams': {
                    'embedding_dim': EMBEDDING_DIM,
                    'hidden_dim':    HIDDEN_DIM,
                    'n_layers':      N_LAYERS,
                    'bidirectional': BIDIRECTIONAL,
                    'dropout':       DROPOUT,
                    'max_len':       MAX_LEN,
                },
            }, CHECKPOINT)
        print(f"  ✓ Checkpoint saved (val_acc={val_acc:.4f})")
# --- Final test evaluation ---
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        preds       = model(batch_x)
        test_loss  += criterion(preds, batch_y).item()
        test_correct += (torch.round(preds) == batch_y).sum().item()
        test_total   += batch_y.size(0)
test_loss /= len(test_loader)
test_acc   = test_correct / test_total
print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
print(f"Best Val Acc: {best_val_acc:.4f}")
    
# --- Sample predictions ---
def predict(text, model, vocab, max_len=300):
    model.eval()
    tokens = tokenize(text)
    ids    = [vocab.get(t, 1) for t in tokens]
    ids    = ids[:max_len] + [0] * (max_len - len(ids))
    tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(tensor).item()
    label = "Positive 😊" if prob > 0.5 else "Negative 😞"
    oov   = [t for t in tokens if t not in vocab]
    print(f"Text  : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"Score : {prob:.4f}  ->  {label}")
    if oov:
        print(f"  OOV : {oov[:10]} -> mapped to <UNK>")
    print()


print("\n── Sample predictions ──────────────────────────────────────\n")
samples = [
        "This latest Bollywood romantic drama starring Ananya Panday and Lakshya is drawing mixed reviews. Some praise the film for mining the relatable, often poignant grey areas of modern relationships, while others feel the execution turns the story into a messy, aesthetically-lit headache with uneven chemistry",
        "A beloved drama that routinely tops fan-voted favorite movie lists. It follows the life of a banker wrongly convicted of murder who spends two decades in a notoriously brutal prison",
        "For a mind-bending sci-fi action experience, Christopher Nolans blockbuster about thieves who infiltrate the subconscious minds of their targets is unmatched.",
    ]
for text in samples:
    predict(text, model, vocab, max_len=MAX_LEN)