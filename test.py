from torch_responder.seq2seq import Seq2SeqResponder
import torch_responder
import os

def main():
    if os.path.exists("responder_s2s.model"):
        model = Seq2SeqResponder()
        model = model.load("responder")
    else:
        model = Seq2SeqResponder()
    while True:
        print("BOT >" + model.predict_from_sentences([input("USER> ")], noise_gain=0.2)[0])

if __name__ == '__main__':
    main()
    