import tensorflow as tf
from keras import layers
import numpy as np
import DataSet_Ph3 as ds_ph3

#Important:
#Tasks: Change the structure of the models
#       Change the training loop (Batchsize, epochs, etc.)
#       Look after the training Dataset if it is correct

# Generator
def build_generator(input_shape):
    model = tf.keras.Sequential()
    # Input layer of the generator
    model.add(layers.InputLayer(input_shape=input_shape))
    # Hidden layers of the generator
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    # Output layer of the generator, padding='same' ensures that the output shape is the same as the input shape
    model.add(layers.Conv2D(1, (3, 3), padding='same', activation='tanh'))
    return model

# Discriminator
def build_discriminator(input_shape):
    model = tf.keras.Sequential()
    # Input layer of the discriminator
    model.add(layers.InputLayer(input_shape=input_shape))
    # Hidden layers of the discriminator
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    # Output layer of the discriminator
    model.add(layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid'))
    return model

# Training loop
def train_gan(generator, discriminator, gan, noisy_music_spec, music_spec_data, epochs, batch_size):
    for epoch in range(epochs):
        # len(noisy_music_spec)// batch_size is how many batches are possible with the given batch_size and the length of the gaven data
        for batch in range(len(noisy_music_spec)// batch_size):
            # Noisy spectrograms
            #there will be created a array with the size of the batch_size. So every batch will have their own data
            noisy_spec = noisy_music_spec[batch * batch_size:(batch + 1) * batch_size]
            music_spec = music_spec_data[batch * batch_size:(batch + 1) * batch_size]

            # Generate denoised spectrograms with the generator
            generated_spec = generator.predict(noisy_spec)

            # Create labels for the discriminator (real data: 1, generated data: 0)
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            # Train the discriminator on real and generated data separately
            d_loss_real = discriminator.train_on_batch(music_spec, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_spec, fake_labels)

            # Compute the combined loss for the generator
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator to deceive the discriminator
            g_loss = gan.train_on_batch(noisy_spec, real_labels)

            # Output the training results
            print(f"Epoch: {epoch + 1}/{epochs}, Batch: {batch + 1}/{len(noisy_music_spec) // batch_size}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

    print("Training finished!")
    return generator, discriminator

# Load the dataset and get the input shape from the Data_loader (DataSet_Ph3.py)
music_spec_data, noisy_music_spec, input_shape = ds_ph3.get_dataset()

print('Values of the dataset are loaded')
# these values should be tested, maybe they are not the best values
epochs = 50
batch_size = 32

# Create Generator and Discriminator
generator = build_generator(input_shape)
discriminator = build_discriminator(input_shape)

# Compile models
generator.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

# Create GAN Model
gan_input = tf.keras.layers.Input(shape=input_shape)
generated_spec = generator(gan_input)
gan_output = discriminator(generated_spec)

gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# Train the GAN
generator, discriminator = train_gan(generator, discriminator, gan, noisy_music_spec, music_spec_data, epochs, batch_size)
# Save the models
print('Saving Models')
generator.save('generator_model_Ph3.h5')
print('Generator Model is saved')
discriminator.save('discriminator_model_Ph3.h5')
print('Discriminator Model is saved')


