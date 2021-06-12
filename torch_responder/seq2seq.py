import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from . import utils as utils
from . import attention

class LSTMEncoder(nn.Module):
    # bi-lstm Encoder
    def __init__(self, vocab_size=8000, embedding_dim=200, hidden_dim=256, num_layers=6, padding_idx=3):
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.5, bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        output, state = self.lstm(x)
        h, c = state
        h = torch.sum(torch.stack(torch.split(h, 2, dim=0)), 1, keepdim=False)
        c = torch.sum(torch.stack(torch.split(c, 2, dim=0)), 1, keepdim=False)
        return output, (h, c)

class LSTMDecoder(nn.Module):
    # lstm-decoder ( with attention )
    def __init__(self, vocab_size=8000, embedding_dim=200, hidden_dim=256, num_layers=6, padding_idx=3):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.5, bidirectional=False)
        self.attn = attention.Attention(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt, state):
        tgt = self.embedding(tgt)
        output, state = self.lstm(tgt, state)
        output = self.attn(src, output)
        output = self.linear(output)
        return output, state

class Seq2SeqModule(nn.Module):
    # Seq2SeqModule
    def __init__(self, vocab_size=8000, embedding_dim=200, hidden_dim=256, padding_idx=3, num_layers=6):
        super(Seq2SeqModule, self).__init__()
        self.encoder = LSTMEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, padding_idx=padding_idx, num_layers=num_layers)
        self.decoder = LSTMDecoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, padding_idx=padding_idx, num_layers=num_layers)

    def forward(self, x, tgt):
        output, state = self.encoder(x)
        decoder_output, _ = self.decoder(output, tgt, state)
        return decoder_output


# SentencePieceを組み込んだクラス。
class Seq2SeqResponder:
    def __init__(self, vocab_size=8000, embedding_dim=200, hidden_dim=256, padding_idx=3, num_layers=6, sentence_length=30):
        self.s2s = Seq2SeqModule(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, padding_idx=padding_idx, num_layers=num_layers)
        self.sp = spm.SentencePieceProcessor()
        self.sentence_length = sentence_length
        self.padding_idx = padding_idx

    # モデルセーブ
    def save(self, file_prefix):
        torch.save(self.s2s, f"{file_prefix}_s2s.model")
    
    # ログファイルからsentencepieceの学習を行う。
    def train_spm_from_log_file(self, src_file_path, file_prefix, char_coverage=0.9995, vocab_size=8000):
        spm.SentencePieceTrainer.Train(
            f"--input={src_file_path}, --model_prefix={file_prefix}_sp --character_coverage={char_coverage} --vocab_size={vocab_size} --pad_id=3"
        )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f"{file_prefix}_sp.model")

    # ログファイルからseq2seqの学習を行う。
    def train_s2s_from_log_file(self, log_file_path, file_prefix, batch_size=100, num_epoch=50, num_steps=100, initial_lr=1., gamma=0.5):
        with open(log_file_path, "r") as f:
            data = f.read()
        data = data.split("\n")
        data = [ self.preprocess_sentence(s) for s in data]
        train_x = data[:-1]
        train_y = data[1:]
        self.train_s2s(train_x, train_y, file_prefix, batch_size=batch_size, num_epoch=num_epoch, num_steps=num_steps, initial_lr=initial_lr, gamma=gamma)
    
    # seq2seqを学習する。
    def train_s2s(self, train_x, train_y, file_prefix, batch_size=100, num_epoch=50, num_steps=100, initial_lr=1., gamma=0.5):
        logger = utils.Logger()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_batch, output_batch = utils.train2batch(train_x, train_y, batch_size=batch_size)
        input_batch = [np.array(a) for a in input_batch]
        output_batch = [np.array(a) for a in output_batch]

        self.s2s.train()
        optimizer = optim.Adam(self.s2s.parameters(), lr=initial_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)
        criterion = nn.CrossEntropyLoss().to(device)
        self.s2s.to(device)
        for step in range(num_steps): # STEP LOOP
            for epoch in range(num_epoch): # EPOCH LOOP
                self.s2s.train()
                epoch_loss = 0
                progress = tqdm(range(len(input_batch)))
                for i in progress: # BATCH ROOP
                    self.s2s.zero_grad()
                    input_tensor, tgt_tensor = torch.LongTensor(input_batch[i]).to(device), torch.LongTensor(output_batch[i]).to(device)

                    src_tensor = self.tgt2src(tgt_tensor)
                
                    output_tensor = self.s2s(input_tensor, src_tensor)

                    loss = 0
                    for j in range(input_tensor.size(1)):
                        loss += criterion(output_tensor[:, j, :], tgt_tensor[:, j])
                    epoch_loss += loss.item() / input_tensor.size(1)

                    loss.backward()
                    optimizer.step()
                    status = f"| EPOCH #{epoch} BATCH #{i} | LR: {round(scheduler.get_last_lr()[0], 5)} | LOSS: {round(loss.item() / input_tensor.size(1), 5)} |"
                    logger.log(status, print_log=False)
                    progress.set_description_str(desc=status, refresh=False)
                avg_epoch_loss = epoch_loss / len(input_batch)
                # === デバッグ用の処理。
                with open("./samples.txt", "r") as f:
                   input_samples = f.read().split("\n")
                print(self.predict_from_sentences(input_samples))
                # ===
                logger.log("-"*60)
                logger.log(f"| EPOCH #{epoch} summary | LR: {round(scheduler.get_last_lr()[0], 5)} | Avg. LOSS: {round(avg_epoch_loss,5)} |")
                logger.log("-"*60)
                logger.save(f"{file_prefix}_log.txt")
            scheduler.step()
            self.save(file_prefix)
            logger.log("="*60)
            logger.log(f"STEP #{step}, Model is saved successfully.")
            logger.log("="*60)
        logger.save(f"{file_prefix}_log.txt")

    # ログファイルからsentencepieceとseq2seq両方を学習させる。
    def train_from_log_file(self, log_file_path, save_path_prefix, batch_size=100, num_epoch=50, num_steps=100, initial_lr=1., gamma=0.5):
        if not os.path.exists(f"{save_path_prefix}_sp.model"):
            self.train_spm_from_log_file(log_file_path, save_path_prefix)
        else:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(f"{save_path_prefix}_sp.model")
        self.train_s2s_from_log_file(log_file_path=log_file_path, batch_size=batch_size, file_prefix=save_path_prefix, num_epoch=num_epoch, num_steps=num_steps, initial_lr=initial_lr, gamma=gamma)

    def predict_with_batch(self, input_data, batch_size=500, noise_gain=0.0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.s2s.eval()
        self.s2s.to(device)
        input_batch = utils.test2batch(input_data, batch_size=batch_size)
        results = []
        with torch.no_grad():
            for i in range(0, len(input_batch)):
                batch_tmp = []
                input_tensor = torch.LongTensor(input_batch[i]).to(device)
                src_tensor = torch.full((input_tensor.size(0), 1), self.padding_idx+5).to(device)
                output, state = self.s2s.encoder(input_tensor)
                h, c = state
                h += (torch.randn(h.size()).to(device) - 0.5) * noise_gain
                c += (torch.randn(c.size()).to(device) - 0.5) * noise_gain
                state = (h, c)
                decoder_hidden = state
                for j in range(0, input_tensor.size(1)):
                    decoder_output, decoder_hidden = self.s2s.decoder(src_tensor, decoder_hidden)
                    decoder_output = torch.argmax(decoder_output, dim=2)
                    src_tensor = decoder_output
                    batch_tmp.append(decoder_output)
                batch_tmp = torch.cat(batch_tmp, dim=1)
                results.append(batch_tmp)
            results = torch.cat(results)
        return results

    def predict_from_sentences(self, sentences, batch_size=500, noise_gain=0.0):
        input_data = np.array([self.preprocess_sentence(s) for s in sentences]) 
        results = self.predict_with_batch(input_data, batch_size=batch_size, noise_gain=noise_gain).tolist()
        return [ self.sp.DecodeIdsWithCheck(s) for s in results ]

    def summary(self):
        return summary(self.s2s)

    # 生成のソースとなるデータにするため、１文字分右にずらして、最初の文字をpaddingにする。
    def tgt2src(self, output):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.cat((torch.full((output.size(0), 1), self.padding_idx).to(device), output), axis=1)
        return t[:, :-1] # right shift
    
    # 前処理(トークン分割)
    def preprocess_sentence(self, sentence):
        ids = self.sp.EncodeAsIds(sentence)[:self.sentence_length-1]
        while len(ids) < self.sentence_length:
            ids.append(self.padding_idx)
        return ids

    def load(self, file_prefix, map_location=None):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f"{file_prefix}_sp.model")
        self.s2s = torch.load(f"{file_prefix}_s2s.model", map_location=map_location)
        return self