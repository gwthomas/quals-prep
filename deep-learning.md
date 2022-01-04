$$
\newcommand{\softmax}{\operatorname{softmax}}
$$



# Deep Learning

## Convolutional neural networks

## Autogregressive models

## Recurrent neural networks

LSTM are designed to allow long-term dependencies. The cell state is computed as
$$
c_t = f_t \circ c_{t-1} + i_t \circ \tilde{c}_t
$$
where $f_t = \sigma(\cdot)$ is the forget gate, $i_t = \sigma(\cdot)$ is the input gate, and $\tilde{c}_t = \tanh(\cdot)$ is the candidate value.

Then we compute the output gate $o_t = \sigma(\cdot)$, and $h_t = o_t \circ \tanh(\cdot)$.

Gate $g$ is compute according to $\sigma(W_g[x_t, h_{t-1}] + b_g)$.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/640px-LSTM_Cell.svg.png)

## Attention and Transformers

![img](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

Each encoder block of a transformer consists of self-attention followed by an MLP (which is applied separately to each position of the input).

Each position of the input has an associated query $q_i$, key $k_i$, and value $v_i$. The attention score between query $i$ and key $j$ is $q_i \cdot k_j$. Transformers use "scaled dot-product attention"
$$
A(Q,K,V) = \softmax\left(\frac{QK\T}{\sqrt{d_k}}\right)V
$$
The matrices $Q$, $K$, and $V$ are computed as $Q = XW^Q$, $K = XW^K$, and $V = XW^V$.

## Generative adversarial networks

Two-player game between a generator $G$ and a discriminator $D$. Typically $G$ is parameterized as a deterministic function of a random "latent" variable $z$, and $D : \X \to [0,1]$ gives a probability $D(x)$ that $x$ is real.  

The discriminator is tasked with distinguishing real samples from fake ones, and the generator is trained to fool the discriminator:
$$
\min_G \max_D \E_{x \sim p^*}[\log D(x)] + \E_{z}[\log(1-D(G(z)))]
$$
We can backpropagate through the discriminator to get gradients for $G$. At equilibrium, the discriminator cannot distinguish the real from the fake data, and (hopefully) the distribution induced by $G$ approximates $p^*$.
