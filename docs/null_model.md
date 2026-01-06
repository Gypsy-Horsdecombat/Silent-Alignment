Null Model for Silent Alignment v2.2
Purpose

This document defines the null model for the Silent Alignment v2.2 protocol.

The null model specifies the expected distribution of semantic similarity and surface similarity in the absence of meaningful alignment, coordination, or task-induced structure. It provides the baseline against which all observed silent alignment results must be interpreted.

Silent Alignment makes no claims without comparison to this null.

Core Question

How often do independently generated language model outputs exhibit semantic convergence above threshold ε while remaining surface-divergent, purely by chance?

The null model answers this question.

Null Hypothesis (H₀)

Under the null hypothesis:

Semantic similarity between outputs arises solely from:

shared training distributions,

generic linguistic structure,

statistical properties of the embedding space.

No alignment is attributable to:

task structure,

relational stabilization,

prompt-induced convergence,

or interaction effects.

Any observed silent overlap under H₀ is treated as baseline coincidence.

Construction of the Null Model
1. Response Pool

Let:

𝑌
=
{
𝑦
1
,
𝑦
2
,
…
,
𝑦
𝑛
}
Y={y
1
	​

,y
2
	​

,…,y
n
	​

}

be a pooled set of model-generated responses produced across multiple distinct tasks, prompts, or runs.

No two responses in a null pairing originate from the same task prompt.

2. Random Pairing

Randomly sample pairs:

(
𝑦
𝑖
,
𝑦
𝑗
)
(y
i
	​

,y
j
	​

) where 
𝑖
≠
𝑗
i

=j

with no shared task, instruction, or contextual linkage.

This random pairing destroys:

task-level semantic coherence,

relational constraint,

prompt-induced convergence.

3. Measurement

For each random pair, compute:

Semantic similarity

Cosine similarity of embeddings

Using the same fixed embedding model as the main protocol

Surface similarity

String similarity metric identical to the main protocol

All metrics, thresholds, and preprocessing steps are identical to those used in experimental runs.

4. Null Distribution

Repeat random pairing 
𝐾
K times to produce empirical distributions:

𝑆
null
=
{
semantic_sim
𝑘
}
S
null
	​

={semantic_sim
k
	​

}

𝐿
null
=
{
surface_sim
𝑘
}
L
null
	​

={surface_sim
k
	​

}

This yields:

Expected semantic similarity distribution under chance

Expected surface similarity distribution under chance

Silent Overlap Under the Null

A null silent overlap occurs when a random pair satisfies:

semantic similarity ≥ ε

surface similarity ≤ τ

The null silent overlap rate is:

𝑃
null
(
silent overlap
)
=
#
{
null pairs satisfying criteria
}
𝐾
P
null
	​

(silent overlap)=
K
#{null pairs satisfying criteria}
	​


This rate defines the baseline frequency of silent alignment absent task structure.

Interpretation Discipline

Observed silent alignment is meaningful only if:

The experimental silent overlap rate significantly exceeds the null rate

Or the observed semantic similarity lies in the extreme tail of 
𝑆
null
S
null
	​


No individual overlap is interpreted in isolation.

What the Null Model Does Not Assume

The null model does not assume:

independence of training data

absence of shared linguistic priors

absence of statistical bias in embeddings

These factors are intentionally included, making the null conservative.

What the Null Model Controls For

The null explicitly controls for:

embedding space geometry

generic topical similarity

stylistic coincidence

language-wide semantic clustering

Any deviation beyond this baseline must be explained by experimental structure, not chance.

Relationship to Experimental Results

All experimental results must be reported alongside:

null distributions,

null overlap rates,

relative position of observed statistics.

Silent Alignment v2.2 makes no claims without a null comparison.

Status

This null model definition is:

fixed,

non-adaptive,

version-locked to Silent Alignment v2.2.

Any changes require a version increment.
