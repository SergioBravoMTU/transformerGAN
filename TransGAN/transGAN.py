import os
import tensorflow as tf
from LSTM_Discriminator import lstm_Discriminator
import dataProcessing as dp
import numpy as np 
import time
import random
import transformerGenerator as tg
import plotlib as pl

dataset_dir = os.path.dirname("Reddit/processed/datasetReduced/total/")
models_dir = os.path.dirname("models/")
gen_dir = os.path.join(models_dir, "generator/")
disc_dir = os.path.join(models_dir, "discriminator/")
gan_dir = os.path.join(models_dir, "gan/")
results_dir = os.path.dirname("results/")
genResults_dir = os.path.join(results_dir, "generator/")
discResults_dir = os.path.join(results_dir, "discriminator/")
ganResults_dir = os.path.join(results_dir, "gan/")
strategy = tf.distribute.get_strategy()

# def sample_gumbel(shape, eps=1e-20): 
#   """Sample from Gumbel(0, 1)"""
#   U = tf.random_uniform(shape,minval=0,maxval=1)
#   return -tf.log(-tf.log(U + eps) + eps)

# def gumbel_softmax_sample(logits, temperature): 
#   """ Draw a sample from the Gumbel-Softmax distribution"""
#   y = logits + sample_gumbel(tf.shape(logits))
#   return tf.nn.softmax( y / temperature)

# temp near 0 for categorical features
def gumbel_softmax(x, vocab_size, temp=1e-10):
    #Distribution
    g = -tf.math.log(-tf.math.log(tf.random.uniform([vocab_size])))
    y = x + g
    return tf.nn.softmax(y/temp)


def gan(g_model, d_model, optimizer, vocab_size, name="gan"):

    d_model.trainable = False
    inputs = tf.keras.Input(shape=(None,))
    dec_inputs = tf.keras.Input(shape=(None,))
    g_output = g_model([inputs, dec_inputs])
    gumbel = gumbel_softmax(g_output, vocab_size)
    
    # softmax = softargmax(gumbel)
    # We need to change in other way as argmax has no gradient
    # argmax_output = tf.keras.layers.Lambda(tf.math.argmax, arguments={'axis':2}, trainable=False)(g_output)
    # softmax = layers.Dense(256, activation=gumble_argmax)(g_output)
    # argmax = tf.math.argmax(gumbel, 2)
    # argmax = Argmax(2)(gumbel)
    # y = tf.reshape(gumbel,[None,200])
    d_output = d_model(gumbel)
    model = tf.keras.Model(inputs = [inputs, dec_inputs], outputs = d_output, name=name)
    model.compile(optimizer, "sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

	def __init__(self, embed_dim, warmup_steps=4000):
		super(CustomSchedule, self).__init__()

		self.embed_dim = embed_dim
		self.embed_dim = tf.cast(self.embed_dim, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps**-1.5)

		return tf.math.rsqrt(self.embed_dim) * tf.math.minimum(arg1, arg2)

# # Get fake examples from the generator
def generate_random_fake_samples(n_samples, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH):

    def generateFakeString(tokenizer, START_TOKEN, END_TOKEN):

        nWords = random.randint(1, MAX_LENGTH-2) 
        # take of tokens from the possibilities
        nSubwords = tokenizer.vocab_size - 2
        # We have to eliminate the End and Start TOKENS
        subwords = [random.randint(1, nSubwords) for subword in tokenizer.subwords if not subword in [START_TOKEN, END_TOKEN]]
        return random.choices(subwords, k = nWords)    

    X = [generateFakeString(tokenizer, START_TOKEN, END_TOKEN) for i in range(n_samples)]
    
    tokenized_X = []
    for sentence in X:
        if len(sentence) <= MAX_LENGTH:
            tokenized_X.append(sentence)
        
    # pad tokenized sentences
    tokenized_X = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_X, maxlen=MAX_LENGTH, padding='post')
   
    y = np.zeros((len(tokenized_X), 1), dtype=np.int64)

    return tokenized_X, y

def generate_fake_samples(g_model, X, n_samples, tokenizer, START_TOKEN, END_TOKEN, maxlen, yOnes=False):
    
    # We have to have two different datasets (one for validation and one for training)
    # obtain n_samples input utterances to predict the outcome
    X = random.choices(X, k=n_samples)
    # Get the prediction of the generation model 
    predX = [tg.predictTokenized(g_model, sentence, tokenizer, START_TOKEN, END_TOKEN, maxlen) for sentence in X]    
    
    # print(predX)
    # pad tokenized sentences
    predX = tf.keras.preprocessing.sequence.pad_sequences(
        predX, maxlen=maxlen, padding='post')
    # print(predX)

    if yOnes:
        y = np.ones((len(predX), 1), dtype=np.int64)
    else:
        y = np.zeros((len(predX), 1), dtype=np.int64)

    return predX, y

def generate_fake_samples_with_dec_input(g_model, X, y, n_samples, tokenizer, START_TOKEN, END_TOKEN, maxlen, yOnes=False):
    
    # We have to have two different datasets (one for validation and one for training)
    index = np.random.randint(0, len(y), n_samples)

    y = np.array(y)[index]
    X = np.array(X)[index]

    dec_inputs = []

    for sentence in y:
        # tokenize sentence
        sentence = START_TOKEN + tokenizer.encode(sentence) 
        # check tokenized sentence max length
        if len(sentence) <= maxlen:
            dec_inputs.append(sentence)

    # pad tokenized sentences
    dec_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        dec_inputs, maxlen=maxlen, padding='post')


    # Get the prediction of the generation model 
    predX = [tg.predictTokenized(g_model, sentence, tokenizer, START_TOKEN, END_TOKEN, maxlen) for sentence in X]    
    
    # print(predX)
    # pad tokenized sentences
    predX = tf.keras.preprocessing.sequence.pad_sequences(
        predX, maxlen=maxlen, padding='post')
    # print(predX)

    if yOnes:
        y = np.ones((len(predX), 1), dtype=np.int64)
    else:
        y = np.zeros((len(predX), 1), dtype=np.int64)

    return predX, dec_inputs, y

def get_real_samples(y, tokenizer,START_TOKEN, END_TOKEN, n_samples, MAX_LENGTH):
    index = np.random.randint(0, len(y), n_samples)

    samples = np.array(y)[index]

    tokenized_X = []

    for sentence in samples:
        # tokenize sentence
        sentence = START_TOKEN + tokenizer.encode(sentence) 
        # check tokenized sentence max length
        if len(sentence) <= MAX_LENGTH:
            tokenized_X.append(sentence)

    # pad tokenized sentences
    tokenized_X = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_X, maxlen=MAX_LENGTH, padding='post')

    y = np.ones((len(samples), 1), dtype=np.int64)

    return tokenized_X, y


def train_discriminator(d_model, g_model, X, y, batch_size, maxlen, tokenizer, START_TOKEN, END_TOKEN, n_iter=100):

    accs = []

    for i in range(n_iter):

        X_real, y_real = get_real_samples(y, tokenizer, START_TOKEN, END_TOKEN, batch_size, maxlen)
        # print(X_real[0:1])
        X_real = dp.onehotencode(X_real, tokenizer)
        # print(X_real[0:1])
        real_loss, real_acc = d_model.train_on_batch(X_real, y_real)

        X_fake, y_fake = generate_fake_samples(g_model, X, batch_size, tokenizer, START_TOKEN, END_TOKEN, maxlen)
        # print(X_fake[0:1])
        X_fake = dp.onehotencode(X_fake, tokenizer)
        # print(X_fake[0:1])
        fake_loss, fake_acc = d_model.train_on_batch(X_fake, y_fake)

        accs.append([real_loss, real_acc, fake_loss, fake_acc])
        print("Epoch ", i+1, ": Real Acc", real_acc, ": Fake Acc", fake_acc)
    
    d_model.save_weights(os.path.join(models_dir, "discWeightsBest.hdf5"))
    return accs

def train_gan(g_model, d_model, gan_model, X_gan, y_gan, epochs, batch_size, tokenizer, START_TOKEN, END_TOKEN, maxlen, dpath, gpath, ganpath, niters=10):

    dLoss = []
    ganLoss = []
    inx = []
    counter = 0

    for i in range(epochs):
        print(i)
        for j in range(niters):
            print(j, batch_size)

            # Train discriminator
            X_real, y_real = get_real_samples(y_gan, tokenizer, START_TOKEN, END_TOKEN, batch_size, maxlen)
            X_fake, y_fake = generate_fake_samples(g_model, X_gan, batch_size, tokenizer, START_TOKEN, END_TOKEN, maxlen)

            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            # print(X_real.shape, y_real.shape)
            # print(X_fake.shape, y_fake.shape)

            print(X.shape)
            print(y.shape)
            X = dp.onehotencode(X, tokenizer)
            d_loss = d_model.train_on_batch(X, y)
            print(X.shape)
            print(y.shape)            
            # Obtain other fake samples (But y labels will be one)
            X_gan_fake, y_dec_input, y_gan_fake = generate_fake_samples_with_dec_input(g_model, X_gan, y_gan, batch_size, tokenizer, START_TOKEN, END_TOKEN, maxlen, yOnes=True)

            print(X_gan_fake.shape, y_dec_input.shape, y_gan_fake.shape)
            # Train GAN model
            gan_loss = gan_model.train_on_batch([X_gan_fake, y_dec_input], y_gan_fake)
            print(d_loss, gan_loss)

            dLoss.append(d_loss)
            ganLoss.append(gan_loss)

            inx.append(counter)
            counter += 1
    d_model.save_weights(dpath)
    g_model.save_weights(gpath)
    gan_model.save_weights(ganpath)
    return dLoss, ganLoss

if __name__ == "__main__":

    embed_dim = 256
    units = 512
    num_heads = 8
    num_layers = 2
    ff_dim = 32
    maxlen = 200
    dropout = 0.1
    test = False
    # File of the dataset
    nAns = 1
    # N epochs we want to train
    epochs = 20
    # Settings for loading and saving the Generator
    nEpochsGenLoad = 35                          # Last to load
    nEpochGen = nEpochsGenLoad + epochs         # Save in this file
    genLoad = True
    # Settings for loading and saving the Discriminator
    nEpochsDiscLoad = 5                         # Last to load
    nEpochDisc = nEpochsDiscLoad + epochs       # Save in this file
    discLoad = False
    # Settings for loading and saving the GAN
    nEpochGanLoad = 10                           # Last to load
    nEpochGan = nEpochGanLoad + epochs          # Save in this file
    ganLoad = False
    # Batch size for Gan training
    batch_size = 128
    # N iters per epoch
    niters = 5
    # EPOCHS TRAINED GEN
    train_disc = False
    train_gen = False
    test_gen = False
    train_gan_model = False
    gen_responses = True
    input_queries = [   'ladies of reddit, what can a dad do to help ensure his daughters have good teen years that aren’t too awkward or embarrassing?' 
                        ,'what’s the most disturbing thing you have ever seen happen in public?'
                        ,'reddit, what’s the best thing you did for your health, be it physical, or even spiritual?'
                        ,'what’s something you should know how to do at your age that you can’t?'
                        ,'if we lived in a society where money didn’t exist and had robots that took care of everything (cultivating, crafting, etc), what would you do in your life?'
                        ,'animals are now as smart as the average human. which animal are you now terrified of?'
                        ,'reddit, if life was a game, how would you review it?'
                        ,'what is your best weight loss method?'
                        ,'what does france do better than other countries?'
                        ,'what fictional villain would u kill if u had the chance to?'
                        ,'what song is stuck in your head?'
                        ,'What are your best fighting tips?'
                    ]

    X, y = dp.readDatafile(os.path.join(dataset_dir, "total_{}_nAns.json".format(nAns)))

    if test:
        X = X[0:100]
        y = y[0:100]    

    # Define start and end token to indicate the start and end of a sentence
    tokenizer, START_TOKEN, END_TOKEN = dp.getStartAndEndTokens(X, y)
    
    vocab_size = tokenizer.vocab_size + 2

    BATCH_SIZE = 64 * strategy.num_replicas_in_sync
    BUFFER_SIZE = 20000

	# Divide the dataset into two different sets 
	# One will be for the normal Generator train, 
	# The other one will be for training the gan.

    genDataset = {}
    ganDataset = {}

    X_gen, X_gan, y_gen, y_gan = dp.divideDataset(X, y, 0.6, 0.2)

    print(X_gen[0:10])
    print(y_gen[0:10])

    X_gen, y_gen = dp.tokenize_and_filter(X_gen, y_gen, tokenizer, START_TOKEN, END_TOKEN, maxlen)

    d_model = lstm_Discriminator(maxlen=maxlen, vocab_size=vocab_size)
    g_model = tg.transformer(
		vocab_size=vocab_size,
		num_layers=num_layers,
		units=units,
		d_model=embed_dim,
		num_heads=num_heads,
		dropout=dropout
    )
    learning_rate = CustomSchedule(embed_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    g_model.summary()
    d_model.summary()
    
    d_model.compile(optimizer, "sparse_categorical_crossentropy", metrics=["accuracy"])

    d_weightsfile_save = os.path.join(disc_dir, "disc_{}_WeightsBest_{}_epochs.hdf5".format(nAns, nEpochDisc))
    d_weightsfile_load = os.path.join(disc_dir, "disc_{}_WeightsBest_{}_epochs.hdf5".format(nAns, nEpochsDiscLoad))
    g_weightsfile_save = os.path.join(gen_dir, "gen_{}_WeightsBest_{}_epochs.hdf5".format(nAns, nEpochGen))
  #  g_weightsfile_save = os.path.join(gen_dir, "gen_{}_WeightsBest_{}_epochs_tr_w_gan_dataset.hdf5".format(nAns, nEpochGen))

    g_weightsfile_load = os.path.join(gen_dir, "gen_{}_WeightsBest_{}_epochs.hdf5".format(nAns, nEpochsGenLoad))
    # g_weightsfile_load = os.path.join(gen_dir, "gen_{}_WeightsBest_{}_epochs_tr_w_gan_dataset.hdf5".format(nAns, nEpochsGenLoad))
    gan_weightsfile_save = os.path.join(gan_dir, "gan_{}_WeightsBest_{}_epochs.hdf5".format(nAns, nEpochGan))
    gan_weightsfile_load = os.path.join(gan_dir, "gan_{}_WeightsBest_{}_epochs.hdf5".format(nAns, nEpochGanLoad))

    if train_disc: 
        start = time.time()
        print("train_disc")
    
        accs = train_discriminator(d_model, g_model, X, y, batch_size, maxlen, tokenizer, START_TOKEN, END_TOKEN, epochs)
        end = time.time() - start
        print("Execution Time:", end)
        
        fname = os.path.join(discResults_dir, "discTrainValues.csv")

        pl.writeAccuracies(not genLoad, fname, accs)
        pl.generateAccsPlots(fname)   
    
    elif train_gen:
        print("train_gen")
        if genLoad:
            g_model.load_weights(g_weightsfile_load)  
        if not test_gen:
		#CHANGE THIS FOR DIFF DATASETS
            X_gan, y_gan = dp.tokenize_and_filter(X_gan, y_gan, tokenizer, START_TOKEN, END_TOKEN, maxlen)
            dataset = dp.getDatasetTensor(X_gan, y_gan, BUFFER_SIZE, BATCH_SIZE)
            g_model.compile(optimizer=optimizer, loss=tg.loss_function, metrics=[tg.accuracy])
            checkpoint = tf.keras.callbacks.ModelCheckpoint(g_weightsfile_save, monitor="loss", mode="min", save_best_only=True, save_weights_only=True, verbose=1)
            hist = g_model.fit(dataset, epochs=epochs, callbacks=[checkpoint])
            fname = os.path.join(genResults_dir, "gen_{}_TrainValues_gan.csv".format(nAns))
            pl.writeHistory(not genLoad, fname, hist)
        else:
            output = tg.predict(g_model, "what fictional villain would u kill if u had the chance to?", tokenizer, START_TOKEN, END_TOKEN, maxlen)
            print(output)
            output = tg.predict(g_model, "what song is stuck in your head?", tokenizer, START_TOKEN, END_TOKEN, maxlen)
            print(output)
            output = tg.predict(g_model, "What are your best fighting tips?", tokenizer, START_TOKEN, END_TOKEN, maxlen)
            print(output)

    elif train_gan_model:
        print("train_gan")
        
        if genLoad:
            g_model.load_weights(g_weightsfile_load)
        if discLoad:
            d_model.load_weights(d_weightsfile_load)

        gan_model = gan(g_model, d_model, optimizer, vocab_size)
        if ganLoad:
            gan_model.load_weights(gan_weightsfile_load)
        
        start = time.time()
        
        dLoss, ganLoss = train_gan(g_model, d_model, gan_model, X_gan, y_gan, 
                    epochs, batch_size, tokenizer, START_TOKEN, END_TOKEN, maxlen, d_weightsfile_save, g_weightsfile_save, gan_weightsfile_save, niters=niters)
        end = time.time() - start
        print("Execution Time:", end)    

        ganResFile = os.path.join(ganResults_dir, "gan_TrainValues")
        discResFile = os.path.join(ganResults_dir, "disc_TrainValues")

        pl.writeAccuracies(not ganLoad, ganResFile, ganLoss)
        pl.writeAccuracies(not discLoad, discResFile, dLoss)
    elif gen_responses:
        if genLoad:
            g_model.load_weights(g_weightsfile_load) 
            for query in input_queries:
                output = tg.predict(g_model, query, tokenizer, START_TOKEN, END_TOKEN, maxlen)
                print(output)

        if ganLoad:
            gan_model = gan(g_model, d_model, optimizer, vocab_size)
            gan_model.load_weights(gan_weightsfile_load) 
            model = tf.keras.Model(inputs=gan_model.input, 
                                    outputs=gan_model.get_layer("transformer").output)
            model.compile(optimizer=optimizer, loss=tg.loss_function, metrics=[tg.accuracy])
            for query in input_queries:
                output = tg.predict(model, query, tokenizer, START_TOKEN, END_TOKEN, maxlen)
                print(output)
    else:
        print("No train option chosen")
