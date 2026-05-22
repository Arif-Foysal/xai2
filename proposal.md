# Concept Lattices as Canonical Feature Hierarchies: Formal Concept Analysis as a Training-Free Foundation for Mechanistic Interpretability

**A Phase 2 Research Proposal**

---

## Abstract

Sparse autoencoders (SAEs) have become the dominant tool for discovering interpretable features inside transformer language models, but they suffer from well-documented problems: hyperparameter sensitivity, training instability, feature splitting and absorption, and seed-dependent irreproducibility. This proposal develops a parallel framework grounded in **Formal Concept Analysis** (FCA) — a branch of order-theoretic mathematics built on Galois connections between objects and attributes — and argues that it can serve as a *canonical, deterministic, low-tuning* counterpart to SAEs. The framework is developed along two parallel tracks: **Track A** applies FCA directly to discretized neural activations via interordinal scaling; **Track B** applies FCA on top of features already discovered by an SAE, treating the SAE as a candidate-atom generator and FCA as the algebraic organizer. We (i) prove that, given a fixed conceptual scaling, the concept lattice is uniquely determined by the activation matrix; (ii) define lattice-theoretic *structural analogues* of polysemanticity and superposition and pre-register the empirical claim that they track the operational notions on toy models with known ground truth; (iii) prove a Galois-embedding theorem connecting SAE-discovered features to join-irreducibles of an extended concept lattice; and (iv) establish order-invariance of the framework under strictly monotone activation transforms. All experiments are designed to run on a laptop (toy models, MNIST MLPs) or a single GPU (GPT-2 small, Pythia-70M).

---

## 1. Motivation and Problem Statement

Modern mechanistic interpretability operates on a simple loop: train an SAE on the activations of some layer, hope its sparse features are monosemantic, and inspect them. The empirical successes have been substantial (Bricken et al. 2023; Templeton et al. 2024), but the methodology has four persistent issues:

**1. Reproducibility.** Two SAEs trained on the same activations with different seeds produce noticeably different feature dictionaries. "Discovered feature" becomes a method-dependent rather than model-dependent property.

**2. Hyperparameter dependence.** Sparsity penalty, dictionary size, learning rate, and architecture (gated, top-k, JumpReLU, transcoder, crosscoder) all change the discovered features. "Feature splitting" — increasing dictionary size finds *new* features that were not absent before but were combined — has no principled stopping criterion.

**3. Continuous-versus-discrete mismatch.** Activations are continuous, but interpretable concepts are discrete categories. SAEs leave this gap unresolved, forcing arbitrary thresholding for downstream analysis.

**4. No canonical structure.** SAE features are vectors in activation space with no algebraic relationships imposed by the method itself. Any hierarchy ("this feature is a subfeature of that one") must be discovered post-hoc.

These are inherent to learned, gradient-based decomposition methods. We propose a complementary, order-theoretic framework that retains a categorical skeleton of the representation while making no use of gradient descent.

## 2. Prior Work and Positioning

The application of FCA to neural systems is not new, and the proposal is explicit about this.

**Endres & Földiák (NeurIPS 2008)** applied FCA to neurophysiological recordings from cortical area STSa, interpreting biological neural codes. They did not study artificial neural networks in the modern sense.

**Hanika & Hirth (2022) — "Conceptual Views of Neural Networks"** is the closest prior work. They use conceptual scaling on last-hidden-layer activations of classification networks to obtain global, descriptive views amenable to subgroup discovery. Our technical delta is sharp: they use conceptual scaling for *global descriptive interpretation of classification logits*; we use it as a *structural decomposition of internal representations* in transformer residual streams, and we prove correspondence theorems linking the resulting lattice to SAE-discovered features.

**Pattern structures (Ganter & Kuznetsov, 2001; Kaytoue et al., 2011)** generalize FCA to non-binary data and are the natural home for continuous activations. They have not been applied to mechanistic interpretability of transformers. We treat pattern structures as an alternative formulation studied in parallel to interordinal scaling (§3.4).

**Fuzzy FCA (Bělohlávek, 2004)** offers a probabilistic relaxation of the incidence relation. We do not develop this track here but flag it in §11 as a follow-up.

**Subsequent applied work** (e.g., MDPI 2024 on medical CNNs; ceur-ws 2019 on black-box explanation) treats FCA as an auxiliary explanation layer atop a network, not as a structural framework for feature discovery in the mechanistic-interpretability sense.

**Contribution of this proposal:**
1. Repositioning FCA for the post-2023 mechanistic-interpretability paradigm: residual streams, transformers, head-to-head comparison with SAEs.
2. A pair of formal results (canonicity given scaling; Galois embedding of SAE features) and two definitions (structural polysemanticity; superposition index) that formalize central concepts of interpretability in order-theoretic terms.
3. A two-track methodology (FCA-on-activations and FCA-on-SAE-features) that addresses the discretization problem honestly rather than hiding it.
4. Empirical validation on toy models with known ground truth and on open SAEs trained on small transformers, with pre-registered falsifiable predictions.

We do **not** claim to invent the FCA-on-NNs idea. We claim to give it foundational mathematical content for an interpretability subfield that did not exist when the prior work was written.

## 3. Mathematical Framework

### 3.1 Formal contexts from neural activations

Let $\mathcal{D} = \{x_1, \ldots, x_n\}$ be a dataset of *contextualized positions* (for transformers, (prompt, token-position) pairs; for MLP/image models, individual inputs) and let $a : \mathcal{D} \to \mathbb{R}^d$ be the activation function for a fixed layer of a fixed model. Write $a_j(x_i)$ for the activation of neuron $j$ on input $x_i$.

A **formal context** is a triple $\mathbb{K} = (G, M, I)$ where $G$ is a set of "objects," $M$ is a set of "attributes," and $I \subseteq G \times M$ is a binary incidence relation. In Track A we take $G = \mathcal{D}$ and let $M$ be derived from neurons via scaling.

**Remark (object correlation).** For transformer activations the objects $\mathcal{D}$ are correlated across positions within a prompt. This does not affect the algebraic content of the lattice but matters for any statistical claim about extents. We will report results both at the position level and aggregated to the prompt level.

### 3.2 Conceptual scaling: discretization as a modeling choice

The non-trivial design choice is how to discretize continuous activation values. We adopt **interordinal scaling** (Ganter & Wille 1999, Ch. 1) as the default:

For each neuron $j$ and each threshold $\tau$ in some finite set $T_j \subseteq \mathbb{R}$, create two attributes $m_{j,\tau}^{\leq}$ and $m_{j,\tau}^{\geq}$, with
$$(x_i, m_{j,\tau}^{\geq}) \in I \iff a_j(x_i) \geq \tau, \qquad (x_i, m_{j,\tau}^{\leq}) \in I \iff a_j(x_i) \leq \tau.$$

**The discretization trilemma.** Three desiderata pull against each other:

- *Losslessness*: taking $T_j$ to be the set of distinct observed values recovers the activation matrix exactly, but gives $|M| = 2d|T_j|$ attributes (e.g., $\approx 7.7 \times 10^6$ for GPT-2 small on 5000 tokens with $d=768$).
- *Tractability*: a small fixed $T_j$ (quantiles at $\{0.25, 0.5, 0.75\}$, or activation/no-activation) keeps the lattice computable.
- *Canonicity*: the more $T_j$ is determined by the data alone, the less the framework depends on tuning.

We cannot maximize all three. The honest move is to report results across a small ladder of scaling choices:

- **S0:** binary scaling — fire/no-fire at a per-neuron quantile (typically 0 for ReLU activations).
- **S1:** ternary or 3-quantile interordinal scaling — fixed $|T_j| = 3$.
- **S2:** full lossless interordinal scaling with iceberg pruning.

Canonicity claims (§4) are made *given* a choice; we do not pretend the choice is free.

### 3.3 The Galois connection and the concept lattice

Given $\mathbb{K} = (G, M, I)$, define the derivation operators
$$A' = \{m \in M \mid \forall g \in A, (g,m) \in I\}, \qquad B' = \{g \in G \mid \forall m \in B, (g,m) \in I\}.$$

The pair $(\cdot)'$ forms a **Galois connection** between $2^G$ and $2^M$ ordered by inclusion. A **formal concept** is a pair $(A, B)$ with $A' = B$ and $B' = A$. The set $\mathfrak{B}(\mathbb{K})$ of formal concepts, ordered by $(A_1, B_1) \leq (A_2, B_2) \iff A_1 \subseteq A_2$, forms the **concept lattice** of $\mathbb{K}$. By the basic theorem of FCA (Wille 1982), $\mathfrak{B}(\mathbb{K})$ is a complete lattice whose meet and join are uniquely determined by $\mathbb{K}$.

### 3.4 Track B: FCA on SAE features

A pre-trained SAE produces a sparse activation pattern for each feature. Treating each feature as a binary attribute (fires above threshold or not) and each object as an input gives a formal context whose discretization is inherited from the SAE rather than chosen. Track B sidesteps the discretization trilemma at the cost of importing the SAE's own design choices. We study both tracks: Track A is the more ambitious claim, Track B is the more defensible empirical pipeline, and the Galois embedding theorem (§4) is the bridge between them.

### 3.5 Pattern structures (alternative formulation)

For Track A we will additionally formulate the framework using **pattern structures** $(G, (\mathbf{D}, \sqcap), \delta)$ on interval patterns (Kaytoue et al. 2011), which avoid scaling by working directly with interval-valued descriptions. This gives a continuous-data version of the lattice and is the natural setting for a future fuzzy/probabilistic extension. We report on it as a sanity check on the scaling-dependence of Track A.

## 4. Theoretical Contributions

We restate the formal content as one theorem, three propositions, and two definitions. The previous draft's "five theorems" overstated content that is either tautological given the setup (canonicity, monotone-invariance) or empirically contingent (polysemanticity, superposition counts).

### Proposition 1 (Canonicity given scaling)

*Fix a scaling regime $S \in \{S0, S1, S2\}$. Given the activation matrix, the formal context $\mathbb{K}_S$ is uniquely determined, and consequently so is its concept lattice $\mathfrak{B}(\mathbb{K}_S)$ up to isomorphism. No random seeds, training procedures, or learned hyperparameters appear.*

The honest content of this proposition is that the only choices entering the lattice are (i) the activation matrix and (ii) the scaling regime — neither of which is a *learned* artifact. Two researchers running the pipeline with the same $S$ on the same activations get the same lattice on the nose.

### Proposition 2 (Order-invariance under monotone transforms)

*Let $\phi : \mathbb{R} \to \mathbb{R}$ be strictly monotone-increasing, applied componentwise to activations. For scalings $S0$ and $S1$ when thresholds are defined as quantiles, and for $S2$ throughout, $\mathfrak{B}(\mathbb{K}_S)$ is invariant under $\phi$.*

**Sketch.** Interordinal scaling with quantile thresholds depends only on the rank order of activation values, which monotone transforms preserve.

### Definition 1 (Structural polysemanticity)

*A neuron $j$ is **structurally polysemantic at scale $S$** if its attribute concepts $\gamma(m_{j,\tau})$ are join-reducible in $\mathfrak{B}(\mathbb{K}_S)$ for the threshold(s) used.*

We do **not** claim that structural polysemanticity is equivalent to the operational notion (multiple human-interpretable senses). The pre-registered empirical claim (§6.3, P2) is that the two notions correlate strongly on toy models with planted ground truth, and that the structural notion can serve as a *training-free proxy* for polysemanticity detection in larger models.

### Definition 2 (Superposition index)

*The **superposition index** of layer activations at scale $S$ is*
$$\sigma_S(\mathbb{K}) \;=\; \frac{|J(\mathfrak{B}(\mathbb{K}_S))|}{\mathrm{rank}(A)},$$
*where $J(\cdot)$ denotes the set of join-irreducible concepts and $A$ is the activation matrix.*

Birkhoff's representation theorem identifies $|J|$ as the minimal number of generators of the lattice; comparing this to the linear rank of $A$ asks whether the model encodes more discrete distinctions than it has linear degrees of freedom. We pre-register (P1) that $\sigma_S > 1$ for layers known to exhibit superposition (toy models with planted features and $d_l < k$) and $\sigma_S \approx 1$ for matched-rank Gaussian baselines.

### Theorem (Galois embedding of SAE features)

*Let $\mathcal{F}_{\mathrm{SAE}}$ be a set of features discovered by an SAE trained on activations underlying $\mathbb{K}_S$, each with a fixed activation threshold inducing a binary attribute on $\mathcal{D}$. Let $\mathbb{K}_S^+ = (G, M \cup \mathcal{F}_{\mathrm{SAE}}, I^+)$ be the extended context. Then:*

1. *The closure operators of $\mathbb{K}_S$ and $\mathbb{K}_S^+$ are related by a Galois adjunction making $\mathfrak{B}(\mathbb{K}_S)$ a sub-$\bigwedge$-lattice of $\mathfrak{B}(\mathbb{K}_S^+)$.*
2. *Each SAE feature whose extent is not the intersection of extents already present in $\mathbb{K}_S$ corresponds to a join-irreducible concept of $\mathfrak{B}(\mathbb{K}_S^+)$.*
3. *The order on SAE features induced by support inclusion is the order induced on the corresponding join-irreducibles by the lattice.*

**Sketch.** (1) Adjoining binary attributes preserves the closure-system structure and refines it; the inclusion of closure systems is the adjunction. (2) An SAE feature's extent that is not already expressible as a meet of existing attribute extents introduces a genuinely new join-irreducible. (3) Direct from the definition of the lattice order.

This is the headline result. It says: (a) SAE features are a *particular selection* from a larger algebraic structure; (b) FCA gives that structure without training; (c) discrepancies between an SAE feature and a join-irreducible of the extended lattice are diagnostic — they identify SAE features that are spurious, redundant, or compositions of more basic features.

**Failure modes (to be characterized empirically in Experiment 4):** SAE features whose extent *is* already an intersection of $\mathbb{K}_S$-extents are *not* new join-irreducibles. We predict these will tend to be low-quality or compositional features by independent autointerp metrics.

### What the framework does not do

To be explicit: FCA discards the linear-geometric structure of activation space. Direction-based interventions (steering vectors, ablation along feature directions) have no immediate FCA analogue; the lattice is a categorical skeleton, not a geometric reconstruction. We treat the framework as a *complement* to SAEs (a ground-truth structural comparator), not a wholesale replacement, and §10 makes this positioning explicit.

## 5. Algorithm and Complexity

Concept-lattice construction uses standard polynomial-delay algorithms:

- **NextClosure** (Ganter, 1984): lexicographic enumeration.
- **AddIntent** (van der Merwe et al., 2004): incremental construction.
- **In-Close** family (Andrews, 2009 onward): currently most efficient for sparse contexts.

Worst-case complexity is $O(|L| \cdot |M|^2 \cdot |G|)$ where $|L|$ is the lattice size. For S0 and S1 on typical activation matrices, $|L|$ is empirically polynomial in $|G| + |M|$. For S2 we rely on **iceberg lattices** (Stumme et al., 2002) with an explicit support threshold $\theta$, bounding the number of extracted concepts by a tunable parameter. We will report raw and pruned lattice sizes for every experiment, and treat $\theta$ as a transparent reported hyperparameter rather than hiding it.

**Realistic scale projections.** For GPT-2 small with $n = 5{,}000$ position-objects and $d = 768$:
- S0 (binary, fire/no-fire): $|M| = 768$, contexts are tractable on a laptop.
- S1 (3-quantile interordinal): $|M| \approx 4{,}600$, tractable with In-Close.
- S2 (lossless): $|M| \approx 7.7 \times 10^6$, requires aggressive iceberg pruning; we will report whether useful structure survives at the support thresholds at which it becomes tractable.

If S2 is intractable in practice on the available hardware, we will say so. Track B and pattern structures (§3.5) provide fallbacks that do not require dense interordinal scaling.

## 6. Experimental Plan

### 6.1 Phase 1 — laptop-feasible

**Experiment 1: Toy models of superposition.** Replicate Anthropic's toy model (Elhage et al. 2022): a $d_h \to d_l \to d_h$ autoencoder with $d_h > d_l$ and known sparse ground-truth features. Compute $\mathfrak{B}(\mathbb{K}_S)$ for $S \in \{S0, S1\}$ on the hidden layer. Test P1 (superposition index) and P2 (structural polysemanticity tracks ground-truth polysemanticity). *Expected runtime: minutes.*

**Experiment 2: MLP on MNIST.** Train a small ReLU MLP. Compute lattices on penultimate-layer activations under S0 and S1. Check whether known digit-stroke features appear as join-irreducibles. Compare against k-means clustering of activations as a non-trivial baseline, and against a small SAE as a learned baseline. *Expected runtime: under an hour.*

**Experiment 3: GPT-2 small, single MLP layer (Track B primary, Track A secondary).** Sample ~5000 (prompt, position) pairs from a curated corpus. Track B: take the published OpenAI / Neuronpedia SAE features for the chosen layer as binary attributes and compute the FCA lattice over them. Track A: same activations under S0 and S1. Compare. *Expected runtime: a few hours on a laptop for Track B; Track A depends on lattice size.*

### 6.2 Phase 2 — single GPU

**Experiment 4: Pythia-70M, head-to-head SAE comparison.** Train an SAE from scratch on the same activations used for FCA Track A. Compute the extended context $\mathbb{K}^+$ and test the Galois-embedding theorem in concrete terms: what fraction of SAE features correspond to join-irreducibles of $\mathbb{K}^+$? Characterize the failure modes (do non-irreducible SAE features have lower autointerp scores?). *This is the core empirical validation of the headline theorem.*

**Experiment 5: Reproducibility comparison.** Train 5 SAEs with different seeds; compute FCA lattices on 5 disjoint random halves of the same activation pool. Compare feature-set Jaccard across SAE seeds vs. join-irreducible-set Jaccard across FCA splits. (This is a genuine stability test, not the trivial "deterministic algorithm gives the same output twice.")

**Experiment 6: Order-invariance under transforms.** Apply $\phi(x) = x^3$ and $\phi(x) = \log(1 + e^x)$ to activations; re-train an SAE and re-compute FCA at S0 and S1 (quantile thresholds). Verify Proposition 2.

### 6.3 Pre-registered falsifiable predictions

**P1 (superposition index).** On the toy superposition model with $k$ planted features and $d_l < k$ hidden dimensions, $\sigma_{S1} \geq 1.2$, and matched-rank Gaussian baselines give $\sigma_{S1} \leq 1.05$.

**P2 (structural polysemanticity tracks ground truth).** On the toy model, the set of neurons identified as structurally polysemantic (Definition 1) agrees with the ground-truth set of polysemantic neurons with F1 $\geq 0.8$. On GPT-2 small, at least 50% of SAE features whose autointerp score (Bills et al. scale) is $\geq 0.7$ correspond to join-irreducibles of the extended context $\mathbb{K}^+$ (Track B / Galois-embedding direction).

**P3 (reproducibility).** Across 5 SAE seeds, feature-set Jaccard similarity is $\leq 0.80$. Across 5 disjoint random halves of the activation pool, the join-irreducible-set Jaccard at fixed support threshold is $\geq 0.90$.

**P4 (order-invariance).** Under monotone activation transforms, SAE feature alignment (top-k cosine matching) drops by $\geq 20$%; FCA join-irreducibles at S0/S1 with quantile thresholds are bit-identical.

If two or more of {P1, P2, P3, P4} fail at the stated thresholds, the central claim of the paper is undermined and we will report the negative result.

## 7. Risk Analysis

**Risk 1: Lattice explosion at S2.** Worst-case concept lattice size is exponential and the realistic projection for GPT-2 small is concerning. **Mitigation:** treat S2 as exploratory, anchor the headline claims on S0/S1 and Track B, and report raw and pruned lattice sizes transparently. Acknowledge that the iceberg threshold $\theta$ is a real hyperparameter, not a hidden one.

**Risk 2: Galois embedding may not hold cleanly.** Some SAE features may not correspond to join-irreducibles. **Mitigation:** the theorem is stated with an explicit precondition (extent not already an intersection of existing extents) and the failure cases are themselves the empirical content of Experiment 4. Showing *which* SAE features fail to decompose cleanly is a paper-worthy result.

**Risk 3: Prior work overlap (Hanika & Hirth 2022).** **Mitigation:** the differentiation is technical (global descriptive vs. structural-decomposition; classification logits vs. transformer residual streams; no SAE correspondence theorems in prior work). One paragraph in §2 is sufficient.

**Risk 4: Structural polysemanticity may not track operational polysemanticity.** Definition 1 is a definition, not a claim — but P2 is a falsifiable claim, and it could fail. **Mitigation:** the toy model gives ground truth; if P2 fails there, we will report the negative result and adjust the framework's claims to focus on the Galois-embedding theorem alone.

**Risk 5: Discretization sensitivity.** Different scaling regimes may give materially different lattices. **Mitigation:** the S0/S1/S2 ladder is reported explicitly; pattern structures (§3.5) provide a scaling-free sanity check.

**Risk 6: Geometry is discarded.** FCA throws away linear structure that interpretability tooling depends on. **Mitigation:** position the framework as a categorical complement to SAEs, not a replacement. The follow-up roadmap (§11) explicitly addresses how lattice structure interacts with geometric interventions.

## 8. Timeline (8-week sprint)

- **Week 1.** Lit review finalization. Implement In-Close in NumPy. Toy superposition experiment (Experiment 1).
- **Week 2.** Formalize Propositions 1, 2 and Definitions 1, 2. MNIST experiment (Experiment 2).
- **Week 3.** GPT-2 small Track B (Experiment 3, Track B). Write draft of Sections 1–3.
- **Week 4.** Prove the Galois-embedding theorem in full. Begin Pythia SAE training.
- **Week 5.** Complete Pythia SAE training; Experiment 4 (core validation). Buffer week for the theorem.
- **Week 6.** Reproducibility (5) and order-invariance (6) experiments. All falsifiable predictions tested.
- **Week 7.** Full draft. Pattern-structures sanity check (Experiment 3, Track A backup). Internal review.
- **Week 8.** Polish; submit to NeurIPS / ICML / ICLR depending on cycle.

Two weeks are budgeted for the Galois-embedding theorem because it is the only theorem with non-trivial content; the previous one-week allocation was optimistic.

## 9. Follow-Up Paper Roadmap

This proposal is designed to seed a multi-paper program, not a one-off contribution:

**Paper 2 — Cohomology over the concept lattice.** Place a sheaf of explanations on the concept lattice (extending Proposal 1 from Phase 1) and use sheaf cohomology to measure inconsistency *with respect to the canonical concept structure*. This fixes the experimental-grounding problem of pure sheaf approaches by anchoring the base space to a data-determined lattice.

**Paper 3 — Circuit lattices.** Extend from single-layer activations to joint activations of multiple layers. Develop a lattice on the cross-layer formal context to give a canonical algebraic structure on circuits.

**Paper 4 — Lattice-guided feature steering.** Use the lattice structure to perform feature ablations along join-irreducible directions identified by reading attribute patterns back into activation space. Compare to SAE-feature ablations on standard interpretability benchmarks. This is where the geometry-discarded concern (§4) is directly addressed.

**Paper 5 — Fuzzy and pattern-structure extensions.** Move from binary scaling to fuzzy FCA (Bělohlávek) and to pattern structures with non-trivial similarity operators, producing a fully continuous-data formulation.

## 10. Why This Matters

The current mechanistic interpretability program rests on a methodology — gradient-trained dictionary learning — whose theoretical foundations are thin. A canonical, low-tuning, algebraically structured alternative would not necessarily replace SAEs in practice, but would (i) provide a ground-truth structural comparator against which SAE results can be validated, (ii) formalize the central concepts of polysemanticity and superposition as structural properties amenable to mathematical analysis, and (iii) connect modern interpretability to the mathematics of order theory and lattice theory. The entire framework can be validated on toy models running in seconds, which makes the foundational claims testable at very low cost.

---

*Proposal prepared as Phase 2 of a two-phase research design process. Phase 3 (mathematical derivations and full proofs) will follow on the basis of this scaffold.*
