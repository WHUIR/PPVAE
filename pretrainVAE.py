# code of pretrainVAE
#!-*- encoding=utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.initializers import Ones, Zeros
from myclass import DropConnect, TiedEmbeddingsTransposed, LayerNormalization, MH_GRU, Attention, Position_Embedding

import numpy as np
import random
import sys
import os
import json

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True 
sess = tf.Session(config=config)

pad_token = 0
oov_token = 1
start_token = 2
end_token = 3

# dataset depends on your need
train_path = 'your data'
val_path = 'your data'
test_path = 'your data'

max_len = 17
max_vocab = 10000

# hyper-parameters
dp = 0.2

emb_size = 256
gru_dim = 150
batch_size = 512
latent_dim = 128
nambda = 20
head_num = 8
head_size = [(emb_size+latent_dim) // head_num, (emb_size+2*latent_dim) // head_num, (emb_size+3*latent_dim) // head_num]


train = []
val = []
test = []

with open(train_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().lower().split(' ') 
        train.append(line)

print('train corpus size:', sum([len(d) for d in train]))
sys.stdout.flush()
print('sequences:', len(train))
sys.stdout.flush()

if os.path.exists('yelp-vocab.json'):
    chars,id2char,char2id = json.load(open('yelp-vocab.json'))
    id2char = {int(i):j for i,j in id2char.items()}
else:
    chars = {}
    for lyric in train:
        for w in lyric: 
            chars[w] = chars.get(w,0) + 1

    print('all vocab:', len(chars))
    sys.stdout.flush()

    sort_chars = sorted(chars.items(), key = lambda a:a[1], reverse=True)
    print(sort_chars[:10])
    sys.stdout.flush()
    chars = dict(sort_chars[:max_vocab])

    id2char = {i+4:j for i,j in enumerate(chars)}


    id2char[start_token] = '<BOS>'
    id2char[end_token] = '<EOS>'
    id2char[oov_token] = '<UNK>'
    id2char[pad_token] = '<PAD>'

    char2id = {j:i for i,j in id2char.items()}
    json.dump([chars,id2char,char2id], open('yelp-vocab.json', 'w'))
    
print('vocab size:', len(char2id))
sys.stdout.flush()

with open(val_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().lower().split(' ')
        val.append(line)

with open(test_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().lower().split(' ')
        test.append(line)


def str2id(s, start_end = False):
    ids = [char2id.get(c, oov_token) for c in s]
    if start_end:
        ids = [start_token] + ids + [end_token]
  
    return ids

def padding(x,y,z):
    ml = max_len
    x = [i + [0] * (ml-len(i)) for i in x]
    y = [i + [0] * (ml-len(i)) for i in y]
    z = [i + [0] * (ml-len(i)) for i in z]
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    return x,y, z
    
def train_generator(zhihu_data):
    x = []
    y = []
    z = []
    
    while True:
        np.random.shuffle(data)    
        for d in data:
            if len(d) > (max_len-2):
                d = d[:max_len-2]
                
            d = str2id(d, start_end=True)
            
            x.append(d)
            y.append(d)
            z.append(d[1:])

            if len(x) == batch_size:
                x,y,z = padding(x, y, z)

                yield [x,y,z], None
                x = []
                y = []   
                z = []

def get_batch_num(data):
    x = []
    y = []
    z = []
    bs_num = 0

    for d in data:
        if len(d) > (max_len-2):
            d = d[:max_len-2]
                
        d = str2id(d, start_end=True)

        x.append(d)
        y.append(d)
        z.append(d[1:])

        if len(x) == batch_size:
            x,y,z = padding(x, y, z)

            bs_num += 1
            x = []
            y = []   
            z = []

    return bs_num
    

def sample(preds, diversity=1.0):
    # sample from te given prediction
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

def argmax(preds):
    preds = np.asarray(preds).astype('float64')
    return np.argmax(preds)

def gen(argmax_flag):
    random_vec = np.random.normal(size=(1, latent_dim))

    start_index = start_token #<BOS>
    start_word = id2char[start_index]

    for diversity in [0.5, 0.8, 1.0]:
        for j in range(3):
            print()
            print('----- diversity:', diversity)

            generated = [[start_index]]
            print('----- Generating -----')
            sys.stdout.write(start_word)
            sys.stdout.flush()

            while(end_token not in generated[0] and len(generated[0]) <= max_len):
                x_seq = pad_sequences(generated, maxlen=max_len,padding='post')
                preds = dec_model.predict([x_seq, x_seq, random_vec], verbose=0)[0]
                preds = preds[len(generated[0])-1][3:]
                if argmax_flag:
                    next_index = argmax(preds)
                else:
                    next_index = sample(preds, diversity)

                next_index += 3
                next_word = id2char[next_index]

                generated[0] += [next_index]
                sys.stdout.write(next_word+' ')
                sys.stdout.flush()
            print()


train_bs_num = get_batch_num(train)
val_bs_num = get_batch_num(val)
test_bs_num = get_batch_num(test)

train_gen = train_generator(train)
val_gen = train_generator(val)
test_gen = train_generator(test)

print(train_bs_num, val_bs_num, test_bs_num)
sys.stdout.flush()


#model part

# discriminator
z_in = Input(shape=(latent_dim, ))
z = z_in
z = Dense(latent_dim, activation=None)(z)
z = LeakyReLU()(z)
z = Dense(latent_dim, activation=None)(z)
z = LeakyReLU()(z)
z = Dense(1, use_bias=False)(z)
dis_model = Model(z_in, z)
dis_model.summary()


#encoder part
encoder_input = Input(shape=(max_len, ), dtype='int32')
emb_layer = Embedding(len(char2id), emb_size)
encoder_emb = emb_layer(encoder_input) 
encoder1 = Bidirectional(GRU(gru_dim))

encoder_h = encoder1(encoder_emb)

z_mean = Dense(latent_dim)(encoder_h)
z_log_var = Dense(latent_dim)(encoder_h)

kl_loss = Lambda(lambda x: K.mean(- 0.5 * K.sum(1 + x[0] - K.square(x[1]) - K.exp(x[0]), axis=-1)))([z_log_var, z_mean])

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon

enc_z = Lambda(sampling)([z_mean, z_log_var])
enc_z = Lambda(lambda x: K.in_train_phase(x[0], x[1]))([enc_z, z_mean])

enc_model = Model(encoder_input, [enc_z, kl_loss])
enc_model.summary()


# decoder part
decoder_input = Input(shape=(max_len,), dtype='int32')
decoder_z_input = Input(shape=(latent_dim, ))
decoder_true_output = Input(shape=(max_len,), dtype='int32')


decoder_dense = Dense(emb_size)
dec_softmax = TiedEmbeddingsTransposed(tied_to=emb_layer, activation='softmax')

decoder_emb = emb_layer(decoder_input)
decoder_emb = Position_Embedding()(decoder_emb)
decoder_z = RepeatVector(max_len)(decoder_z_input)
decoder_h = decoder_emb

for layer in range(3):
    decoder_z_hier = Dense(latent_dim, activation=None)(decoder_z)
    decoder_h = Concatenate()([decoder_h, decoder_z_hier])
    decoder_h_attn = Attention(head_num, head_size[layer], max_len)([decoder_h, decoder_h, decoder_h])
    decoder_h = Add()([decoder_h, decoder_h_attn])
    decoder_h = LayerNormalization()(decoder_h)
    decoder_h_mlp = Dense(head_size[layer]*head_num, activation='relu')(decoder_h)
    decoder_h = Add()([decoder_h, decoder_h_mlp])
    decoder_h = LayerNormalization()(decoder_h)
    decoder_h = Position_Embedding()(decoder_h)


decoder_h = decoder_dense(decoder_h)
decoder_output = dec_softmax(decoder_h)

dec_model = Model([decoder_input, decoder_true_output, decoder_z_input], decoder_output)
dec_model.summary()

# train discriminator or PretrainVAE
enc_in = Input(shape=(max_len, ), dtype='int32')
z_in = Input(shape=(latent_dim, ))
enc_model.trainable = False
dec_model.trainable = False
z_fake, kl_loss = enc_model(enc_in)
z_real_score = dis_model(z_in)
z_fake_score = dis_model(z_fake)

dis_train_model = Model([enc_in, z_in],
                      [z_fake_score, z_real_score])


#WGAN-DIV
#param
k = 2
p = 6

d_loss = K.mean(z_real_score - z_fake_score)

real_grad = K.gradients(z_real_score, [z_in])[0]
fake_grad = K.gradients(z_fake_score, [z_fake])[0]

real_grad_norm = K.sum(real_grad**2, axis=[1])**(p / 2)
fake_grad_norm = K.sum(fake_grad**2, axis=[1])**(p / 2)
grad_loss = K.mean(real_grad_norm + fake_grad_norm) * k / 2

w_dist = K.mean(z_fake_score - z_real_score)

dis_train_model.add_loss(-(d_loss - grad_loss))
dis_train_model.compile(optimizer=Adam(5e-4, 0.0))
dis_train_model.metrics_names.append('w_dist')
dis_train_model.metrics_tensors.append(w_dist)
dis_train_model.metrics_names.append('kl_loss')
dis_train_model.metrics_tensors.append(kl_loss)

# train encoder-docoder of PretrainVAE
enc_model.trainable = True
dec_model.trainable = True
dis_model.trainable = False
enc_z, kl_loss = enc_model(enc_in)
z_fake_score = dis_model(enc_z)

dec_in = Input(shape=(max_len, ))
dec_true = Input(shape=(max_len, ))
dec_true_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(dec_true)
dec_out = dec_model([dec_in, dec_true, enc_z])

xent_loss = K.sum(K.sparse_categorical_crossentropy(dec_true, dec_out)*dec_true_mask[:,:,0])/K.sum(dec_true_mask[:,:,0])
d_loss = K.mean(-z_fake_score)
all_loss = xent_loss + nambda*d_loss

enc_dec_train_model = Model([enc_in, dec_in, dec_true], dec_out)
enc_dec_train_model.add_loss(all_loss)
enc_dec_train_model.compile(Adam(5e-4, 0.0))

enc_dec_train_model.metrics_names.append('ce_loss')
enc_dec_train_model.metrics_tensors.append(xent_loss)
enc_dec_train_model.metrics_names.append('kl_loss')
enc_dec_train_model.metrics_tensors.append(kl_loss)

dis_train_model.summary()
enc_dec_train_model.summary()

def id2str(ids):
    return [id2char[x] for x in ids]

def gen_bs(vec, topk=3):
    """beam search
    """
    print('\nbeam search...')
    sys.stdout.flush()
    xid = np.array([[start_token]] * topk)
    vec = np.reshape(np.array([vec]*topk), (topk, latent_dim))
    scores = [0] * topk
    for i in range(max_len): 
        x_seq = pad_sequences(xid, maxlen=max_len,padding='post')
        proba = dec_model.predict([x_seq, x_seq, vec])
        proba = proba[:, i, 3:]
        log_proba = np.log(proba + 1e-6)
        arg_topk = log_proba.argsort(axis=1)[:,-topk:] 
        _xid = [] 
        _scores = [] 
        if i == 0:
            for j in range(topk):
                _xid.append(list(xid[j]) + [arg_topk[0][j]+3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(topk): 
                    _xid.append(list(xid[j]) + [arg_topk[j][k]+3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:] 
            _xid = [_xid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            yid.append(_xid[k])
            scores.append(_scores[k])
        xid = np.array(yid)

    s = id2str(xid[np.argmax(scores)])
    print(' '.join(s))
    sys.stdout.flush()
    
def reconstruct(num):
    for i in range(num):
        print('\nreconstructing, first false second true')
        s = next(test_gen)[0][0][0]
        s_w =  ' '.join([id2char[x] for x in s])
        print(s_w)
        sys.stdout.flush()
        s_v, _ = enc_model.predict(np.array([s]))
        s_r = gen_from_vec(0.8, s_v, False)
        s_r = gen_from_vec(0.8, s_v, True)
        
def gen_from_vec(diversity, vec, argmax_flag):
    start_index = start_token #<BOS>
    start_word = id2char[start_index]
    print()

    generated = [[start_index]]
    sys.stdout.write(start_word)

    while(end_token not in generated[0] and len(generated[0]) <= max_len):
        x_seq = pad_sequences(generated, maxlen=max_len,padding='post')
        preds = dec_model.predict([x_seq, x_seq, vec], verbose=0)[0]
        preds = preds[len(generated[0])-1][3:]
        if argmax_flag:
            next_index = argmax(preds)
        else:
            next_index = sample(preds, diversity)
        next_index += 3
        next_word = id2char[next_index]

        generated[0] += [next_index]
        sys.stdout.write(next_word+' ')
        sys.stdout.flush()   

def interpolate(diversity, num):
    s1 = next(test_gen)[0][0][0]
    s2 = next(test_gen)[0][0][0]
    
    vec1, _ = enc_model.predict(np.array([s1]))
    vec2, _ = enc_model.predict(np.array([s2]))
    print('interpolate with sampling')
    print(' '.join([id2char[x] for x in s1]))
    sys.stdout.flush()
    for i in range(1, num+1):
        alpha = i/(num+1)
        vec = (1-alpha) * vec1 + alpha * vec2
        gen_from_vec(diversity, vec, argmax_flag=False)
    print(' '.join([id2char[x] for x in s2]))
    sys.stdout.flush()
    
    print('interpolate with argmax')
    print(' '.join([id2char[x] for x in s1]))
    sys.stdout.flush()
    for i in range(1, num+1):
        alpha = i/(num+1)
        vec = (1-alpha) * vec1 + alpha * vec2
        gen_from_vec(diversity, vec, argmax_flag=True)
    print(' '.join([id2char[x] for x in s2]))
    sys.stdout.flush()


iters_per_sample = train_bs_num
total_iter = 300000

best_val = 100000.0
best_result = []

for i in range(total_iter):
    for j in range(3):
        # train D for 3 steps
        z_sample = np.random.normal(size=(batch_size, latent_dim))
        K.set_value(dis_train_model.optimizer.lr, 5e-4)
        d_loss = dis_train_model.train_on_batch(
            [next(train_gen)[0][0], z_sample], None)
    for j in range(1):
        # train encoder-decoder for 1 step
        z_sample = np.random.normal(size=(batch_size, latent_dim))
        K.set_value(enc_dec_train_model.optimizer.lr, 5e-4)
        g_loss = enc_dec_train_model.train_on_batch(next(train_gen)[0], None)
        
    if i % 100 == 0:
        print ('iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss))
        sys.stdout.flush()
        
    if (i % 2000 == 0) and (i != 0):
        gen(False)
        gen(False)
        reconstruct(5)
        for _n in range(5):
            vec =  np.random.normal(size=(1, latent_dim))
            gen_bs(vec, 1)
        print('diversity 0.8')
        sys.stdout.flush()
        interpolate(0.8, 8)
        print('diversity 1.0')
        sys.stdout.flush()
        interpolate(1.0, 8)
    
    if i % 2000 == 0 :
        val_loss = enc_dec_train_model.evaluate_generator(val_gen, steps=val_bs_num)
        print('val loss,', val_loss)
        sys.stdout.flush()
        if val_loss[1] <= best_val:
            best_val = val_loss[1]
            best_result = val_loss
            print('saving weights with best val:', val_loss)
            sys.stdout.flush()
            enc_model.save_weights('pretrain/yelp/encoder-'+str(i)+'.h5')
            dec_model.save_weights('pretrain/yelp/decoder-'+str(i)+'.h5')
            dis_model.save_weights('pretrain/yelp/dis-'+str(i)+'.h5')
    if (i%5000 == 0) and (i!=0) :
        enc_model.save_weights('pretrain/yelp/encoder-'+str(i)+'.h5')
        dec_model.save_weights('pretrain/yelp/decoder-'+str(i)+'.h5')
        dis_model.save_weights('pretrain/yelp/dis-'+str(i)+'.h5')
        







