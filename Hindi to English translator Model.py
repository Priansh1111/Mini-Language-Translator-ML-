# save as train_en_hi_seq2seq.py and run
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Optional: for BLEU evaluation
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu

# ---------------------------
# Config / Hyperparameters
# ---------------------------
CSV_PATH = r"C:\Users\shett\OneDrive\Desktop\English to Hindi Translator\Hindi_English_Truncated_Corpus.csv"
SRC_COL = "english_sentence"
TGT_COL = "hindi_sentence"
MAX_VOCAB_SRC = 20000   # top words for English
MAX_VOCAB_TGT = 20000   # top words for Hindi
EMBED_DIM = 256
ENC_UNITS = 512
DEC_UNITS = 512
BATCH_SIZE = 64
EPOCHS = 30
TEST_SIZE = 0.1
VAL_SIZE = 0.1
MAX_LEN_SRC = 60
MAX_LEN_TGT = 60
MODEL_DIR = "./en_hi_seq2seq"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# Simple text cleaning
# ---------------------------
def clean_text_en(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9?.!,']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def clean_text_hi(text):
    text = text.strip()
    # keep Devanagari and punctuation; remove extraneous ASCII punctuation
    text = re.sub(r"[^\u0900-\u097F0-9?.!,']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(CSV_PATH)
# Keep only rows where both columns exist and are strings
df = df[[SRC_COL, TGT_COL]].dropna().astype(str)
df[SRC_COL] = df[SRC_COL].apply(clean_text_en)
df[TGT_COL] = df[TGT_COL].apply(clean_text_hi)

# Add start and end tokens to target sentences (for decoder training)
START_TOKEN = "<sos>"
END_TOKEN = "<eos>"
df[TGT_COL] = df[TGT_COL].apply(lambda x: f"{START_TOKEN} {x} {END_TOKEN}")

# Optional: filter by reasonable length
df['len_src'] = df[SRC_COL].apply(lambda x: len(x.split()))
df['len_tgt'] = df[TGT_COL].apply(lambda x: len(x.split()))
df = df[(df['len_src'] <= MAX_LEN_SRC) & (df['len_tgt'] <= MAX_LEN_TGT)].copy()

# ---------------------------
# Tokenizers
# ---------------------------
src_tokenizer = Tokenizer(num_words=MAX_VOCAB_SRC, filters="", oov_token="<unk>")
src_tokenizer.fit_on_texts(df[SRC_COL].tolist())

tgt_tokenizer = Tokenizer(num_words=MAX_VOCAB_TGT, filters="", oov_token="<unk>")
tgt_tokenizer.fit_on_texts(df[TGT_COL].tolist())

# Vocabulary sizes (+1 for padding)
SRC_VOCAB = min(MAX_VOCAB_SRC, len(src_tokenizer.word_index) + 1)
TGT_VOCAB = min(MAX_VOCAB_TGT, len(tgt_tokenizer.word_index) + 1)

print(f"Src vocab: {SRC_VOCAB}, Tgt vocab: {TGT_VOCAB}")

# ---------------------------
# Sequences & Padding
# ---------------------------
src_seqs = src_tokenizer.texts_to_sequences(df[SRC_COL].tolist())
tgt_seqs = tgt_tokenizer.texts_to_sequences(df[TGT_COL].tolist())

src_padded = pad_sequences(src_seqs, maxlen=MAX_LEN_SRC, padding='post')
tgt_padded = pad_sequences(tgt_seqs, maxlen=MAX_LEN_TGT, padding='post')

# For training decoder inputs and targets:
# decoder_input: all tokens except last
# decoder_target: all tokens except first
decoder_input = tgt_padded[:, :-1]
decoder_target = tgt_padded[:, 1:]

# Train/val/test split
X_src, X_src_test, X_dec_in, X_dec_in_test, Y_tgt, Y_tgt_test = train_test_split(
    src_padded, decoder_input, decoder_target, test_size=TEST_SIZE, random_state=42
)
X_src_train, X_src_val, X_dec_in_train, X_dec_in_val, Y_tgt_train, Y_tgt_val = train_test_split(
    X_src, X_dec_in, Y_tgt, test_size=VAL_SIZE/(1 - TEST_SIZE), random_state=42
)

print("Shapes:", X_src_train.shape, X_dec_in_train.shape, Y_tgt_train.shape)

# ---------------------------
# Dataset pipeline
# ---------------------------
train_dataset = tf.data.Dataset.from_tensor_slices(((X_src_train, X_dec_in_train), Y_tgt_train))
train_dataset = train_dataset.shuffle(2048).batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices(((X_src_val, X_dec_in_val), Y_tgt_val))
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

# ---------------------------
# Model: Encoder, Attention, Decoder
# ---------------------------
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, enc_output, dec_hidden):
        # enc_output: (batch, seq_len, enc_units)
        # dec_hidden: (batch, dec_units) -> expand to (batch, 1, dec_units)
        dec_hidden_time = tf.expand_dims(dec_hidden, 1)
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(dec_hidden_time))
        score = self.V(score)  # (batch, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len, 1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, enc_units)
        return context_vector, tf.squeeze(attention_weights, -1)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, enc_units):
        super().__init__()
        self.enc_units = enc_units
        self.embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.lstm = layers.Bidirectional(layers.LSTM(enc_units, return_sequences=True, return_state=True))

    def call(self, x):
        x = self.embedding(x)
        enc_out, fw_h, fw_c, bw_h, bw_c = self.lstm(x)
        # combine forward and backward states by concatenation
        state_h = tf.concat([fw_h, bw_h], axis=-1)  # (batch, enc_units*2)
        state_c = tf.concat([fw_c, bw_c], axis=-1)
        return enc_out, state_h, state_c

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, dec_units):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.lstm = layers.LSTM(dec_units, return_state=True, return_sequences=True)
        self.fc = layers.Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self, dec_input, dec_hidden, enc_output):
        # dec_input: (batch, seq_len)
        embedded = self.embedding(dec_input)
        # compute attention using last hidden state
        context_vector, attn_weights = self.attention(enc_output, dec_hidden)
        # context_vector: (batch, enc_units*2?) ensure compatible by projecting if needed
        # Expand context to time dimension and concat with embeddings
        context_expanded = tf.expand_dims(context_vector, 1)
        # optionally repeat to match time steps; we'll concat along features
        repeated_context = tf.repeat(context_expanded, tf.shape(embedded)[1], axis=1)
        lstm_input = tf.concat([embedded, repeated_context], axis=-1)
        output, h, c = self.lstm(lstm_input, initial_state=[dec_hidden, tf.zeros_like(dec_hidden)])
        logits = self.fc(output)
        return logits, h, c, attn_weights

# Instantiate encoder/decoder
encoder = Encoder(SRC_VOCAB, EMBED_DIM, ENC_UNITS // 2)  # because bidirectional doubles
# decoder units should match encoder final hidden size (ENC_UNITS)
decoder = Decoder(TGT_VOCAB, EMBED_DIM, DEC_UNITS)

# ---------------------------
# Loss and optimizer
# ---------------------------
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    # real: (batch, seq_len), pred: (batch, seq_len, vocab)
    mask = tf.cast(tf.math.not_equal(real, 0), dtype=tf.float32)
    loss_ = loss_object(real, pred)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

optimizer = tf.keras.optimizers.Adam()

# ---------------------------
# Training step (teacher forcing)
# ---------------------------
@tf.function
def train_step(inp, targ_inp, targ_real):
    with tf.GradientTape() as tape:
        enc_output, enc_h, enc_c = encoder(inp)
        # Initialize decoder hidden with encoder hidden state (project if sizes differ)
        dec_hidden = enc_h  # shape: (batch, ENC_UNITS)
        # Run decoder over full sequence (teacher forcing: feed targ_inp)
        preds, dec_h, dec_c, _ = decoder(targ_inp, dec_hidden, enc_output)
        loss = loss_function(targ_real, preds)
    variables = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss

# ---------------------------
# Training loop
# ---------------------------
best_val_loss = float('inf')
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    steps = 0
    for (batch, ((inp, dec_inp), targ)) in enumerate(train_dataset):
        loss = train_step(inp, dec_inp, targ)
        total_loss += loss.numpy()
        steps += 1
    train_loss = total_loss / steps if steps else 0.0

    # Validation loss (simple loop)
    val_losses = []
    for ((inp_v, dec_inp_v), targ_v) in val_dataset:
        enc_output_v, enc_h_v, enc_c_v = encoder(inp_v)
        preds_v, _, _, _ = decoder(dec_inp_v, enc_h_v, enc_output_v)
        val_losses.append(loss_function(targ_v, preds_v).numpy())
    val_loss = np.mean(val_losses) if val_losses else 0.0

    print(f"Epoch {epoch}/{EPOCHS} — train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

    # Save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        encoder.save_weights(os.path.join(MODEL_DIR, "encoder_best.h5"))
        decoder.save_weights(os.path.join(MODEL_DIR, "decoder_best.h5"))
        print("Saved best model weights.")

# ---------------------------
# Inference utilities
# ---------------------------
# create index -> word mapping for target
index_to_word_tgt = {i: w for w, i in tgt_tokenizer.word_index.items()}
index_to_word_tgt[0] = "<pad>"
# note: tokenizers are 1-indexed for words

def decode_sequence(input_sentence, max_len=MAX_LEN_TGT):
    # preprocess
    s = clean_text_en(input_sentence)
    seq = src_tokenizer.texts_to_sequences([s])
    seq = pad_sequences(seq, maxlen=MAX_LEN_SRC, padding='post')
    enc_out, enc_h, enc_c = encoder(seq)
    dec_hidden = enc_h
    # start token id
    start_id = tgt_tokenizer.word_index.get(START_TOKEN)
    end_id = tgt_tokenizer.word_index.get(END_TOKEN)
    if start_id is None or end_id is None:
        raise ValueError("Start or end tokens not found in target tokenizer.")
    dec_input = np.array([[start_id]])
    output_sentence = []
    for t in range(max_len - 1):
        preds, dec_h, dec_c, attn = decoder(dec_input, dec_hidden, enc_out)
        # preds shape: (batch=1, seq_len=1, vocab)
        preds = tf.squeeze(preds, axis=0)  # (seq_len, vocab)
        preds_token = tf.argmax(preds[0]).numpy()
        if preds_token == end_id:
            break
        word = tgt_tokenizer.index_word.get(preds_token, "<unk>")
        output_sentence.append(word)
        # prepare next input (feeding predicted token)
        dec_input = np.array([[preds_token]])
        dec_hidden = dec_h
    return " ".join(output_sentence)

# ---------------------------
# Quick BLEU evaluation on test set
# ---------------------------
def bleu_on_test(n_samples=200):
    n = min(n_samples, len(X_src_test))
    sample_idx = np.random.choice(len(X_src_test), n, replace=False)
    scores = []
    for idx in sample_idx:
        src_seq = X_src_test[idx:idx+1]
        # convert back to text
        src_text = " ".join([src_tokenizer.index_word.get(i, "") for i in src_seq[0] if i != 0]).strip()
        pred = decode_sequence(src_text)
        # reference: Y_tgt_test is target tokens (shifted), rebuild reference text by mapping tokens to words
        ref_tokens = [tgt_tokenizer.index_word.get(i, "") for i in Y_tgt_test[idx] if i != 0 and i != tgt_tokenizer.word_index.get(END_TOKEN)]
        ref_text = " ".join(ref_tokens).replace(START_TOKEN, "").strip()
        # BLEU expects tokenized references
        ref_tok = nltk.word_tokenize(ref_text)
        hyp_tok = nltk.word_tokenize(pred)
        try:
            sc = sentence_bleu([ref_tok], hyp_tok, weights=(0.5, 0.5))
        except:
            sc = 0.0
        scores.append(sc)
    return np.mean(scores)

print("Evaluating BLEU on test set (this may take a while)...")
avg_bleu = bleu_on_test(100)
print("Average BLEU (approx):", avg_bleu)

# ---------------------------
# Save tokenizers for later use
# ---------------------------
import pickle
with open(os.path.join(MODEL_DIR, "src_tokenizer.pkl"), "wb") as f:
    pickle.dump(src_tokenizer, f)
with open(os.path.join(MODEL_DIR, "tgt_tokenizer.pkl"), "wb") as f:
    pickle.dump(tgt_tokenizer, f)

print("Training complete. Models and tokenizers saved to", MODEL_DIR)
print("Try decoding: print(decode_sequence('how are you'))")
