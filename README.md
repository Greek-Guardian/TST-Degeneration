# **Time-Series Transformers — The Degeneration of Attention and the Failure of Representation Learning**

> Transformer revolutionized NLP and CV, but failed to conquer time series.
>
> Our study reveals: the problem is **not** the attention mechanism itself—
>
> it lies in the **representation space** where the Transformer block operates,
>
> which is poorly structured from the very beginning.

![](https://picx.zhimg.com/80/v2-58ba87d2a964be82c32a7819f0efdbd7_1440w.png?source=ccfced1a)

---

## 1. Background: When Transformers Meet Time Series

Transformers have achieved tremendous success in **NLP** and  **computer vision** .

From GPT to ViT, the attention mechanism seems nearly omnipotent.

However, when applied to  **time series forecasting** , researchers discovered an awkward fact:

> In many datasets, Transformers perform **no better—or even worse—than simple linear models.**

Despite numerous variants—Informer, Autoformer, FEDformer, PatchTST, iTransformer—the improvements remain marginal and inconsistent.

This leads to a deeper question:

> What exactly goes wrong when Transformers are used for time series?

In our paper 👉 [*Why Attention Fails: The Degeneration of Transformers into MLPs in Time Series Forecasting*](https://arxiv.org/abs/2509.20942),

we provide a systematic answer through both theoretical and empirical analysis.

> We find that the attention module in Transformers contributes little to performance in time series tasks—
>
> the entire model effectively  **degenerates into an MLP** .

---

## 2. Experimental Findings: Attention Is Barely “Working”

A natural way to verify whether attention matters is to **ablate** it.

However, simply removing the attention module drastically reduces FLOPs and parameter count, which confounds fair comparison.

To isolate the true role of attention, we designed a series of **perturbation experiments** of varying strength.

We experimented by:

* Replacing the attention matrix with zeros, averages, or fixed patterns;
* Injecting noise or smoothing into the attention scores of trained models;
* Gradually converting Transformers into MLPs (by increasing patch size until only one token remains);
* Removing positional encoding.

![](https://pica.zhimg.com/80/v2-dcdb8c01946e3f38eb714212f0196306_1440w.png?source=ccfced1a)

Across all settings, results were consistent:

> No matter how we “**perturb**” the attention mechanism, model performance barely changed.

In other words, **FFN (feed-forward network)** layers dominate predictive performance,

while attention layers serve as little more than decorative components—

the Transformer has  **degenerated into an MLP** .

---

## 3. Attention Requires a Structured Latent Space

### 🧠 1. The Premise of Attention

The core of attention lies in computing similarity through the dot product between queries and keys:

[

\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V

]

This formulation assumes that **Q** and **K** reside in a  **meaningful latent space** ,

where geometric relationships (distances and angles) reflect semantic similarity.

If the latent space is chaotic or unstructured, then the dot product ( Q \cdot K ) becomes meaningless—

attention turns into random weighting.

### 🌀 2. Linear Embedding: Wrong from the Start

Ideally, the latent space should form a **high-dimensional manifold** capturing nonlinear dynamics and state transitions of the time series.

However, most time series Transformers employ  **linear or convolutional embeddings** :

* Linear embedding (a single linear projection), or
* CNN embedding (a weight-sharing linear layer).

Both are essentially  **linear transformations** : ( z = Wx + b ).

If ( W ) is full-rank, this transformation is a  **linear isomorphism** —

the latent space is geometrically identical to the input space, up to rotation and scaling.

Hence the contradiction:

> If the input and latent space are linearly isomorphic, what is there left to “learn” in representation learning?

The model learns no new structure; attention in such a **pseudo-latent space** has no meaningful relationships to exploit.

(Even using MLPs or TCNs as embedding does not fundamentally fix this issue, as shown later.)

### 🔬 3. Toy Dataset: Diagnosing a Broken Latent Space

Unlike NLP or CV, where attention maps are interpretable,

time series correlations between patches or channels are often opaque—even to domain experts.

To make attention behavior observable, we designed a **toy dataset** with clear state transitions.

![](https://pica.zhimg.com/80/v2-c809650075cc998cdc5e68a3262765ea_1440w.png?source=ccfced1a)

![](https://pic1.zhimg.com/80/v2-bcbedfddca8686a34ee3c85d92a8df02_1440w.png?source=ccfced1a)

Ideally, the latent space should encode distinct “states,” and attention should focus on key event patches.

Yet we observed the opposite:

> Attention weights were nearly uniform across all patches, showing no preference for meaningful events.

In short, the model **fails to form semantic concepts** of “events” or “states.”

The latent space is unstructured—making attention ineffective.

---

## 4. ViT as a Contrast: Why Transformers Work in Vision

A natural question arises:

> “But ViT also uses linear embeddings—why does it work?”

We conducted the same attention perturbation experiments on Vision Transformers.

![](https://pica.zhimg.com/80/v2-b581d7bef39ca6b6f71d4e12a619bab6_1440w.png?source=ccfced1a)

Findings:

* Perturbing attention in early layers barely affected performance;
* Perturbing later layers caused a significant drop.

Hence:

> The early layers of ViT are also partially “degenerated.”
>
> They primarily assist representation learning, while deeper layers perform true attention in a well-formed latent space.

In other words, early Transformer blocks help  organize the embedding space and learn representations.

only after this structuring does attention begin to operate semantically.

---

## 5. The Inherent Difficulty of Time-Series Representation

One might ask:

> “Why not simply deepen PatchTST, like ViT?”

As shown below, depth alone brings little benefit.

![](https://pic1.zhimg.com/80/v2-a8e4f6fa0f7ed312acca2e5bc2d2bc2a_1440w.png?source=ccfced1a)

Our appendix further analyzes why **representation learning** is intrinsically difficult in time series.

### 1️⃣ Dimensional Expansion

Time series inputs are typically low-dimensional, yet embeddings are forcibly expanded.

For instance, PatchTST maps 16-dimensional inputs to 128D, and iTransformer maps 96D to 512D.

This contradicts the usual ML intuition that dimensional compression aids semantic representation.

> According to the Johnson–Lindenstrauss lemma, a sufficiently large latent space is required to host nearly orthogonal semantic directions.
>
> However, excessively large latent spaces make it difficult for embeddings to learn meaningful geometry—hindering effective QK dot products.

![](https://pic1.zhimg.com/80/v2-568b895f6e731d6e035fa3f7de4ce270_1440w.png?source=ccfced1a)

### 2️⃣ Cross-Domain Heterogeneity

Time series data vary drastically across domains—finance, meteorology, healthcare, traffic—

with little shared statistical structure.

Building “foundation models” like in NLP thus becomes extremely challenging.

### 3️⃣ Embedding Type Is Not the Key

We compared various embedding forms—linear, CNN, MLP, residual, etc.—

and the degeneration phenomenon persisted.

Hence,  **the root cause lies deeper than embedding choice** .

![](https://pic1.zhimg.com/80/v2-918a48866e6adb6e17b2bf36e51fad67_1440w.png?source=ccfced1a)

![](https://picx.zhimg.com/80/v2-524478a61f6c8b0f6d617d67e7ec3b12_1440w.png?source=ccfced1a)

---

## 6. Toward a Solution: Let Attention Work in a Meaningful Space

In the appendix, we discuss potential future directions.

> The key to advancing time-series Transformers lies  **not in redesigning attention** ,
>
> but in  **building better latent representations** .

### 🧩 1. Inspirations from Vision and Speech Models

In image generation, **Latent Diffusion Models (LDM)** introduced a crucial idea:

Transformers do not operate directly on raw pixels but on **latent representations** produced by a VAE.

Similarly, speech-generation models like **Qwen2.5-Omni** and **LLaMA-Omni** adopt **encoder–decoder architectures** to first learn high-quality semantic embeddings before applying attention.

The common principle:

> These models let Transformers operate in  **structured latent spaces** ,
>
> rather than on raw, noisy signals—greatly improving attention efficiency and generalization.

### 🔄 2. Implication for Time Series: Build a “Time-Series Encoder”

Translating this idea, we argue that time series forecasting should first learn a **semantic, structured latent space** via an encoder–decoder framework.

Only then should the Transformer perform attention-based modeling  **within that latent space** .

### 🧱 3. A Promising Direction: VQ-VAE / RQ-VAE–Based Discrete Latent Spaces

A particularly promising approach is to use **VQ-VAE** or  **RQ-VAE** ,

which quantize continuous signals into a  **discrete codebook** .

This effectively converts time series into  **token sequences** ,

perfectly aligned with the Transformer’s discrete input–output paradigm.

During forecasting:

* The encoder maps time series into discrete tokens;
* The Transformer models these token sequences autoregressively;
* The decoder reconstructs continuous signals from predicted tokens.

Thus, Transformers model **semantic tokens** rather than raw fluctuations.

### ⚙️ 4. Advantages of This Paradigm

The **discrete latent space + autoregressive prediction** framework offers multiple benefits:

1️⃣ **Stronger semantic consistency**

Transformers operate in a unified codebook space, immune to scale and noise variations.

2️⃣ **Fewer parameters**

Traditional models like PatchTST require large linear heads to concatenate all tokens for output.

In contrast, discrete-token prediction drastically reduces redundancy.

3️⃣ **Alignment with mainstream Transformer paradigms**

This approach naturally aligns time-series forecasting with standard **autoregressive** frameworks—

mirroring practices in language and image modeling.

---

## 7. Paper and Code

📄 Paper: [*Why Attention Fails: The Degeneration of Transformers into MLPs in Time Series Forecasting*](https://arxiv.org/abs/2509.20942)

💻 Anonymous implementation: [https://github.com/Greek-Guardian/TST-Degeneration](https://github.com/Greek-Guardian/TST-Degeneration)
