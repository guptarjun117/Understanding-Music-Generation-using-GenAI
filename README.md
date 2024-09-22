# Understanding Generative AI
Both TTI and TTM technology fall under the umbrella concept of Generative AI, a subset of Deep Learning models in artificial intelligence. These models are designed to produce new data, such as text, and images, that resemble a given set of training data utilizing discovering trends that usually go undetected by humans (unsupervised training), or by active reinforcement using training data with known results (supervised training). The most common models in this domain include Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and autoregressive models like the Generative Pre-trained Transformer (GPT). These models (whose differences will be discussed in greater detail) work on the same fundamental idea for generating content: Akin to the great artists of the Renaissance like Michelangelo who began their careers copying paintings, they are exposed to lots of samples of the media we want them to recreate (i.e. paintings songs, stories, etc.). This "AI artist" can then create its unique iterations of the content that it has been exposed to, mimicking the styles and distinguishing features of the content it was trained on.
Today, Generative AI has seen exponential growth in popularity and prominence in conjunction with the rapid developments and breakthroughs made in recent years, and models like OpenAI's DALL-E, Midjourney, and Stable Diffusion have made a name for themselves, boasting their adeptness and precision in generating content given limited prompts. But could we somehow harness this same concept for creating music? What about generating a piece of music with a given prompt, like "a calming violin melody backed by a distorted guitar riff"?

# Text-to-Image and Text-to-Music models
## The Middle Ground
A commonality of both models is that they utilize the transformer model with the self-attention mechanism. First introduced in 2017 in the paper "Attention Is All You Need", the idea of Transformers has set a benchmark on how Generative AI can push the boundaries of what AI models can understand and produce.
![image](https://github.com/user-attachments/assets/c59e8ddd-e9a1-4f41-8a9c-42a081a4196c) Transformers Architecture

What makes Transformers so 'state-of-the-art'? It is the sequence-to-sequence (Seq2Seq) Encoder-Decoder architecture and the Attention mechanism. Seq2Seq is a type of neural network that converts one sequence of elements into another. As the name suggests, it's about transforming sequences. Seq2Seq models are composed of two main components: an Encoder and a Decoder. These models excel at tasks like translation, where a sequence in one language is converted into a sequence in another language. The Encoder (the left side of the structure) processes the input sequence and converts it into a higher-dimensional vector (n-dimensional vector). This vector is then fed into the Decoder (the right side of the structure), which produces the output sequence. The attention mechanism decides which parts of an input sequence are crucial at each step. For instance, while reading, one focuses on a word and retains important keywords for context. In the Encoder-Decoder analogy, the attention mechanism would be like the Encoder noting down vital keywords from the input sequence and providing them to the Decoder. These keywords help the Decoder understand the context and importance of different parts of the sequence. Transformer architecture does not use any Recurrent Neural Networks (RNNs) such as GRU or LSTM.

## The Difference
However, it's not just transformers that do the job. Text-to-Image models often utilize Generative Adversarial Networks (GANs) to generate images meanwhile Text-to-Music models often use Variational Autoencoders (VAEs) to generate music.
## GANs
![image](https://github.com/user-attachments/assets/27ee5437-a979-425a-8a54-5231e962e0e6) GAN Architecture

Also introduced in the paper "Attention Is All You Need" in 2017, GAN models operate in operates in an unsupervised manner where two neural networks, the generator and the discriminator, compete to improve their predictions. The generator, a convolutional neural network, aims to produce outputs that resemble real data. The discriminator, a deconvolutional neural network, tries to distinguish between real and artificially created outputs. The generator starts by sampling some noise, typically using a normal or uniform distribution. This noise, conceptually representing latent features of the images (like color and shape), is then used to produce an image. The discriminator, on the other hand, is trained using both real and generated images. It aims to recognize features that make images appear real and provides feedback to the generator to produce more realistic images. The training process involves alternating steps, with the generator aiming to produce images that the discriminator can't distinguish from real ones. Over time, the GAN model converges to produce natural-looking images. An essential aspect of GANs is backpropagation. The discriminator outputs a value indicating the likelihood that an image is real. The objective is to maximize the chance of recognizing real images as real and generated images as fake. The generator's goal is to produce images that fool the discriminator. Both networks are trained alternately until the generator produces high-quality images.

![image](https://github.com/user-attachments/assets/c39b380e-0514-4405-a2fb-1685e6446814)
![image](https://github.com/user-attachments/assets/8d44ed0a-e528-41e7-bfc2-39093dea57d3)

GANs have a wide range of applications. For instance, they can transform a horse image into a zebra, and produce photorealistic representations of human faces. In the realm of video production, GANs can model human behavior within frames, predict subsequent video frames, or even create deep fakes.

## VAEs

![image](https://github.com/user-attachments/assets/663fe97f-e98f-49de-8913-f63de28419d7) VAE Architecture

VAEs also work very similarly to what we just learned, they also possess an encoder and decoder architecture. It transforms real samples into an ideal data distribution using an encoder network. This data distribution is then passed to a decoder network to produce generated samples. If the generated samples are close enough to the real samples, an autoencoder model is trained. The factor that makes VAE unique is the Gaussian mixture model (GMM) where the goal is to adjust the encoder and decoder such that the Evidence Lower Bound (ELBO) is maximized. Maximising ELBO is equivalent to minimizing the KL divergence between q(z|x) and P(z) and maximizing the reconstruction error. The encoder and decoder are adjusted iteratively. Each time the decoder improves, the encoder is adjusted to match it.

# MusicLM by Google

![image](https://github.com/user-attachments/assets/876e70a2-bb8c-4a93-a478-c6f5be0d92f7) MusicLLM Architecture

One of the greatest TTM examples that can be given right is MusicLM by Google. Introduced in early 2023, MusicLM revolutionized the field of TTM. By leveraging a clever combination of deep learning-based models, MusicLM generates convincing short music clips with good audio fidelity. Looking at the structure above, MusicLM combines 3 models: SoundStream, w2v-Bert, and MuLan. Check out some examples created here.
## Soundstream
Soundstream is a neural audio codec or an audio compressor in simple terms which is able to take a waveform and compress it at a lower bit rate. Since this is a neural audio codec, it maintains a high reconstruction quality and it is able to compress and decompress audio to maintain audio Fidelity up to 24 kHz. Soundstream has been trained with a residual Vector quantization.
## w2v-BERT
w2v-BERT is a BERT-based LLM model specifically trained for audio on speech for semantic information. Meaning it just not only takes into account the syntactic approach (the grammatical structure of the text) but also the meaning behind it (semantics). In NLP, the BERT (Bidirectional Encoder Representations from Transformers) is a Transformer model by Google that is very versatile and can trained to a specific case. In this case, the BERT was trained specifically for audio and was named w2v-BERT. Some other BERT variants include RoBERTa, DistilBERT, SBERT, BERTSUM, etc.
## MuLan
Published by Google as well, MuLan is a Music-Text joint embedding model that links music to its description in a free-form format using embedding. Music-Text model combines two embeddings: Audio and Text into a joint embedding that is a single Vector that has information both about audio and its textual information or description. MuLan as a whole is trained on pairs of music clips and their corresponding text annotations so that the model can have information for both: the audio and the text related to it. The Text-embedding network is a BERT model, pre-trained on text. The Audio-embedding network is the ResNet-50 model, which is a variant of the ResNet (Residual Network) architecture, a deep convolutional neural network (CNN) designed for image classification and other vision tasks. The "50" in ResNet-50 refers to the number of layers in the network.
## Training Steps
![image](https://github.com/user-attachments/assets/ffab7e66-18f7-46c4-83a3-f1a9de2594f4) MusicLM architecture expanded and annotated

1. The first step is to extract tokens from the input:
- Acoustic tokens (A) with SoundStream
- Semantic tokens (S) with w2v-BERT
- Audio tokens (Ma) with MuLan
2. The RVQ and k-means are used for post-processing for the embeds that come out of these models to quantize them.
3. Predict the semantic tokens conditioned on MuLan; audio tokens: Ma → S.
4. Predict the acoustic tokens conditioned on MuLan audio tokens and semantic tokens: (Ma, S) → A.

To understand from a big picture point of view, recall what we learned about transformers earlier about their nature and structure. Here, each stage is modeled autoregressively as a Seq2Seq task using decoder-only Transformers.
What is an autoregression model? Let's understand by looking at the math for the semantic and the acoustic modeling.

Semantic Modelling: p (St | S < t, MA)
Acoustic Modelling: p (At | A < t, S, MA)

A probability distribution where it gets the semantic tokens at a given time step t that depends on all of the semantic tokens at previous steps or time steps. For Acoustic, getting the acoustic tokens at a given step t depends on all the acoustic and semantic tokens at previous steps along. This is the basic concept of an autoregressive model. Along with autoregression we also have the conditioning in MuLan.
The acoustic model captures the fine-grained acoustic details, while the semantic model improves long-term music structure and adheres to text descriptions.

# Inference
![image](https://github.com/user-attachments/assets/dc2b350f-9f72-4043-b824-32577d384444)

1. Provide a text prompt
2. Extract MuLan text token Mt
3. Predict semantic tokens conditioned on MuLan text tokens
4. Predict acoustic tokens conditioned on MuLan tokens and semantic tokens
5. Reconstruct audio passing the acoustic tokens to the SoundStream decoder

# Future aspects, impact, and applications
After reading the article you have now gained an understanding of how Generative AI works, the predominant LLM Generative AI models that are taking the AI space by storm, and how the boundaries of Generative AI are being pushed in 2023, especially in the Music generation. Probably 10 years ago an average human would've never thought of the possibility of music creation by just giving a text prompt. But here we are!

So what now? The use case and impact of TTM is endless. Some applications include:
- Functional Music: Type of music where the primary goal isn't musicality but rather achieving a specific end goal, like relaxation or focus. Used by marketing departments in any company for background music.
- Video Game OSTs: How AI can generate dynamic soundtracks that change based on in-game events, narratives, and visual experiences.
- Automatic Bandmates: The idea of AI models trained on famous musicians' styles, allowing solo artists to have virtual bandmates.
- Music Exercise Generation: The potential of AI in generating exercises for musicians, like guitar tablatures or sheet music, based on existing pieces.

Yet, like all innovations, this comes with its set of challenges. There are pressing concerns about originality, potential copyright infringements, and the looming threat of job losses within the music industry. Given the nascent stage of TTM technology, its reproducibility remains limited. This places smaller entities, such as universities and startups, at a disadvantage, struggling to compete with AI powerhouses like Google and OpenAI. However, the narrative should emphasize collaboration over replacement. There's immense potential for AI to augment, not supplant, human musicians, fostering a harmonious blend of technology and artistry.

# Reference
- https://medium.com/r/?url=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762
- https://medium.com/r/?url=https%3A%2F%2Fjunyanz.github.io%2FCycleGAN%2F
- https://medium.com/r/?url=https%3A%2F%2Fstats.stackexchange.com%2Fquestions%2F512242%2Fwhy-does-transformer-has-such-a-complex-architecture
- https://medium.com/r/?url=https%3A%2F%2Fgoogle-research.github.io%2Fseanet%2Fmusiclm%2Fexamples%2F
- https://medium.com/r/?url=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fgoogleai%2Fmusiccaps
