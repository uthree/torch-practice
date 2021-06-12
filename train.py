from torch_responder.seq2seq import Seq2SeqResponder
import torch_responder
import os

def main():
    if os.path.exists("responder_s2s.model"):
        model = Seq2SeqResponder()
        model = model.load("responder")
    else:
        model = Seq2SeqResponder(num_layers=6, sentence_length=24, vocab_size=10000)
    print(model.summary())
    model.train_from_log_file("F:/logs/nucc_discord.txt", "responder", num_steps=100, num_epoch=10, initial_lr=1e-05, batch_size=500 ,gamma=0.9)

if __name__ == '__main__':
    main()
    