Inspiration
We can't draw. We use machine learning. We create modern art.

What it does
We use fundamental concepts of a generative adversarial network (GAN) to use GPUs to generate artwork. A GAN consists of two neural networks: a generator network and a discriminator network. The generator network generates noise that resembles a training set of pictures. The discriminator networks compares the training set pictures and the generated pictures and decides if a generated pictures is "real" or "fake." The "real" pictures are then saved.

How we built it
We followed documentation of opencv to preprocess over eleven thousand training images. We interpolated the images to downscale them into 64 by 64 pixels; then we converted the images into gray scale for faster training time. We made layers for the neural networks with Keras and TensorFlow. We trained the networks on laptop GPUs.

We then feed the generated 64x64 pixel "face" into a scale-up network and then a recolor network online. The networks uses the opensource Keras pre-trained convolutional neural network VGG16.

Challenges we ran into
The original networks was too large to fit into the vRAM of laptop GPUs, so we lowered the batch size of the training program from 128 pictures per batch into 64 pictures per batch to reduce vRAM usage. By doing so, we also increase the training time.

Accomplishments that we're proud of
The network generated pictures that somewhat resembles the human face.

What we learned
It is difficult to work with Tensorflow and CUDA packages on Windows, and it should be easier on Linux. GPUs are much more powerful than CPU in machine learning tasks involving layers and matrix multiplication.

What's next for Neural Network Face Generator
Adding a lambda layer to increase accuracy of the network.

Built With
