# Multi-Pitch Estimation with Nonnegative Matrix Factorization

In this project we will examine how NMF can be used for multi-pitch estimation. 

## Multi-pitch Estimation (MPE)

[Multi-pitch estimation](https://music-ir.org/mirex/wiki/2020:Multiple_Fundamental_Frequency_Estimation_%26_Tracking) is the process of retrieving the active notes at any given point for a given audio signal. Given an input spectrogram $X$ representing a piece of audio containing a musical piece, an MPE model would predict $Y$, the frame-level activations of notes in that piece of audio.

![](res/MPET.png)

Such predictions are useful for general music information retrieval tasks, such as automatic chord estimation, automatic music transcription, genre classification etc.

## Nonnegative Matrix Factorization (NMF)

[Nonnegative matrix factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) is a mathematical technique which decomposes a matrix $X$ into two nonnegative components, $W$ and $H$.

![](res/500px-NMF.png)

NMF has been used for deep learning tasks, where the marices $W$ and $H$ are approximated using by minimizing the following term as a reconstruction loss:

$$
    \Vert X - WH \Vert _F
$$

This way, we can define $W$ as a learnable dictionary, and $H$ as the activation coefficients of $W$. The goal of NMF in machine learning is to create a highly interpretable, lighteweight representation of our data. 

### NMF in MPE

NMF has been specifically used in MPE, where $W$ is the harmonic profile of each note and $H$ represents the activations of such notes. 

![](res/Non-negative-matrix-factorization-NMF-a-A-given-non-negative-matrix-V-is_W640.jpg)

NMF is by default an unsupervised process. However, supervised objectives can be used to influence the initialization and form of the component matrices. Specifically, by encouraging similarity between the activation coefficients $H$ and the frame-level notes, we can enforce the dictionary $W$ to mimic the harmonic profile of each note, thus allowing both cleaner reconstruction and generalization. 

**The objectives of this project are:**

* Create a dictionary matrix $W$ by minimizing:
$$
    \Vert X - WH \Vert_F
$$
where $X$ is our input spectrogram and $H = Y$ is equal to the note activation posteriorgram.

* Use $W$ to approximate $Y$ for new, unseen samples.
* Evaluate this strategy on a simple dataset ([Guitarset](https://guitarset.weebly.com/)) as well as a complex one ([MAESTRO](https://magenta.withgoogle.com/datasets/maestro))
* **ONLY IF I HAVE ENOUGH TIME** expand the methodology for Deep NMF. 