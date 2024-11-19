# Solve inverse problems with score-modelling

This repository is built on top of https://github.com/yang-song/score_sde_pytorch.

[1]: https://arxiv.org/abs/2111.08005

This project extends [Solving Inverse Problems in Medical Imaging with Score-Based Generative Models][1] to support solving any linear inverse problems. 

## Related work

[2]: https://github.com/wyhuai/DDNM
[3]: https://github.com/bahjat-kawar/ddrm

- [Denoising Diffusion Null-Space Models][2]
- [Denoising Diffusion Restoration Models][3]

## Background

A linear inverse problem is a problem of the form

$$
\mathbf{y} = A\mathbf{x} + \epsilon
$$

where $A \in \mathbb{R}^{m \times n}$ is a (known) degradation matrix, $\mathbf{x} \in \mathbb{R}^n$ is the unknown image, $\mathbf{y} \in \mathbb{R}^m$ is the observed features, and $\epsilon$ is the noise.

The original paper suggests decomposing $A$ into $A = \mathcal{P}(\Lambda) T$ where $T \in \mathbb{R}^{n \times n}$ is an invertible transformation and $\mathcal{P}(\Lambda) \in \mathbb{R}^{m \times n}$ is a diagonal matrix values in $\{0,1\}$ that "drops" unused values. 

The core update to apply (before applying the standard diffusion update) is:

$$
\hat{x}'\_{t_i} = T^{-1}[
    \lambda \Lambda \mathcal{P}^{-1}(\Lambda) \hat{y}\_{t_i}
    + (1-\lambda) \Lambda T \hat{x}\_{t_i}
    + (I-\Lambda) T \hat{x}\_{t_i}
]
$$

## Contribution

I found that the quality of this update heavily depends on what $T$ and $\mathcal{P}(\Lambda)$ are chosen (they are not unique). In some cases (e.g., deblurring), it is really hard to come up with a good $T$ and $\mathcal{P}(\Lambda)$ (using FFT deconvolution as $T$ works terribly).

[4]: https://en.wikipedia.org/wiki/Singular_value_decomposition

I found much better results using [Singular Value Decomposition][4] to decompose $A$ into $A = U \Sigma V^T$, where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal matrices and $\Sigma \in \mathbb{R}^{m \times n}$ is a rectangular diagonal matrix. Note 

> [!NOTE]
> Orthogonal matrix is one such that $A^T = A^{-1}$.

We then use $T = V^T$ and $\mathcal{P}(\Lambda) = U \Sigma$. Note this breaks the original setting where $\mathcal{P}(\Lambda)$ is a diagonal matrix. (Todo: What is the significance of this?) $\Lambda$ is now simply a rectangular diagonal matrix like $\Sigma$, except it has values 1 for the corresponding diagonal entries of $\Sigma$ that are non-zero, and 0 otherwise.

The update equation now becomes:

$$
\hat{x}'\_{t_i} = V [
    \lambda \Lambda \Sigma^{-1} U^T \hat{y}\_{t_i} 
    + (1-\lambda) \Lambda V^T \hat{x}\_{t_i} 
    + (I-\Lambda) V^T \hat{x}\_{t_i}
]
$$

### Why does this work?

We can simplify the above equation to:

$$
\begin{align}
\hat{x}'\_{t_i}
    &= V[
        \lambda \Lambda (\Sigma^{-1} U^T \hat{y}\_{t_i}- V^T \hat{x}\_{t_i})
        + V^T \hat{x}_{t_i}
    ] \\
    &= \lambda V \Lambda (\Sigma^{-1} U^T \hat{y}\_{t_i}- V^T \hat{x}\_{t_i})
        + \hat{x}\_{t_i}
\end{align}
$$

Todo: I'm stuck... Why does this work?

### Implementation

$A$ is typically quite large. For example, for CelebA (3 x 256 x 256), A is 196608 x 196608. This is too large to fit in memory and to perform SVD on. 

Instead, most degradation operator can be decomposed into a row-wise linear operator $A_r \in \mathbb{R}^{h \times h'}$, a column-wise linear operator $A_c \in \mathbb{R}^{w \times w'}$, and a channel-wise linear operator $A_{ch} \in \mathbb{R}^{c \times c'}$. For example, Gaussian blurring can be decomposed into a row-wise convolution, a column-wise convolution, and an identity channel operator. The full operator $A$ can then be represented as $A = A_r \otimes A_c \otimes A_{ch}$, where $\otimes$ is the Kronecker product. 

Conveniently, we can perform SVD on each of the operators separately, and then combine them together to form $U$, $\Sigma$, and $V$ for $A$.

$$
\begin{align}
    A_r &= U_r \Sigma_r V_r^T \\
    A_c &= U_c \Sigma_c V_c^T \\
    A_{ch} &= U_{ch} \Sigma_{ch} V_{ch}^T \\
    \\
    U &= U_r \otimes U_c \otimes U_{ch} \\
    \Sigma &= \Sigma_r \otimes \Sigma_c \otimes \Sigma_{ch} \\
    V &= V_r \otimes V_c \otimes V_{ch} \\
    \\
    A 
        &= (U_r \Sigma_r V_r^T) 
        \otimes (U_c \Sigma_c V_c^T)
        \otimes (U_{ch} \Sigma_{ch} V_{ch}^T) \\
        &= (U_r \otimes U_c \otimes U_{ch})
        (\Sigma_r \otimes \Sigma_c \otimes \Sigma_{ch}) 
        (V_r \otimes V_c \otimes V_{ch})^T \\
        &= U \Sigma V^T
\end{align}
$$

This makes the compuation much faster and require significantly less memory.
