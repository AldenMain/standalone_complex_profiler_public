# Standalone Complex Profiler — Public Release

### Profiling the Psychological Substructure of Language without Labels, Dogma, or Surveillance

This repository contains the public-facing architecture of an unsupervised NLP system designed to surface *standalone psychological complexes* from natural language. These complexes are emergent, recursive structures in discourse — often rooted in trauma, ideology, grief, or defensive identity formation. 

The profiler isolates linguistic *why-patterns*, not just *what-claims*. No private data is included. No users are classified. What you see here is the clean scaffold — modular, interpretive, and ready to be wielded.

---

##  Why This Exists

Modern online discourse is a projection theatre. People repeat themselves not because they’re boring — but because something in them is unresolved. Reddit threads, Twitter spats, spiritual blogs, and gender wars: they aren’t just data — they’re symptoms.

The **Standalone Complex Profiler** was built to:
- Detect unconscious psychological patterns in large-scale natural language corpora
- Cluster those patterns without applying diagnostic labels or supervised learning
- Reveal the emergent structure of emotional, ideological, or identity-driven loops
- Do all this **without** extracting, surveilling, or violating individual users

This system doesn’t predict behavior. It doesn’t assign moral scores. It detects structure — and it exposes meaning.

---

##  Core Objective

To transform unstructured text into clusters of psychological significance — using unsupervised methods that resist premature categorization. The pipeline does not group people by topic or opinion, but by the **mechanisms underlying how they speak**, write, and repeat.

This is not sentiment analysis. This is signal analysis.

---

##  What It Does

1. **Extracts psychologically meaningful signal features**  
   Affective polarity, subjectivity, negation, modal verbs, self-reference, projection markers, and more.

2. **Embeds those features**  
   UMAP (Uniform Manifold Approximation and Projection) reduces the vector space while preserving structural topology.

3. **Clusters with HDBSCAN**  
   Density-based clustering reveals naturally emergent groups — no need to predefine how many clusters exist.

4. **Interprets each cluster**  
   Using feature overlays and summary heuristics to generate human-readable insights into what unites the language patterns.

5. **Visualizes the structure**  
   UMAP projections colored by cluster ID, confidence, and outlier scores — making interpretation both visible and falsifiable.

---

##  Epistemological Assumptions

This profiler makes several core bets:

- **Language is performative.** It doesn’t just describe — it discloses.
- **Repetition is structure.** If someone keeps saying the same thing, it’s not content — it’s compulsion.
- **Labels are liabilities.** Predefining categories blinds you to emergent ones.
- **Meaning is relational.** Context matters; anomaly is often signal, not noise.

It assumes that interpretability is a moral obligation — and that unsupervised systems should still answer the question: *“Why did it group this?”*

---

##  Included Modules

This public release includes only safe, generalizable modules. No personal data or scraping logic is exposed.

### `feature_engineering/`
- Signal construction: Extracts linguistic and psychological features from preprocessed text
- Modular and stateless: Works on any well-formed input DataFrame

### `modeling/`
- UMAP: Dimensionality reduction with preservation of relational geometry
- HDBSCAN: Cluster discovery without rigid boundaries
- `run_full_process.py`: Orchestrator script for reproducibility

### `interpretation/`
- Auto-labeling, feature overlays, and human-readable cluster summaries
- Designed for interpretive use, not classification

### `utils/`
- Minimal helper functions: clean abstractions, no fluff

---

##  Tech Stack

- **Python 3.10+**
- `pandas`, `numpy`, `scikit-learn`
- `umap-learn`, `hdbscan`
- `matplotlib`, `plotly`
- Built for clarity, not cleverness

No black boxes. No fragile frameworks. Just data, logic, and structure — as it emerges.

---

##  Philosophy & Methodology

This system is informed by:
- **Complex systems theory**: emergent patterns in noisy systems
- **Depth psychology**: repetition compulsion, projection, identity defenses
- **Unsupervised learning theory**: let the data speak before we categorize it

The profiler doesn’t assume it knows what to look for. Instead, it finds what *keeps happening*, and asks what that repetition reveals.

---

##  Use Cases

- **Digital anthropology**: Understand how ideologies, neuroses, or collective grief form online
- **Content moderation (ethically)**: Surface problematic structures without moral labeling
- **Mental health research**: Detect maladaptive language patterns in public discourse
- **Narrative analytics**: Map the unconscious architecture of storytelling and identity performance
- **AI explainability**: Use unsupervised patterns to retro-engineer "why" from machine behavior

---

##  Ethics, Privacy, and Boundaries

This project exists **in opposition to surveillance culture**. It was designed to be:
- **Non-extractive**: No web scraping, no user IDs, no profiles
- **Interpretation-first**: Not used to classify or rank individuals
- **Transparent**: Built for inspection, critique, and improvement
- **Epistemically humble**: It shows structure — not truth

---

##  Repo Structure
standalone_complex_profiler_public/
│
├── feature_engineering/ # Signal and feature extraction
├── modeling/ # UMAP, HDBSCAN, orchestrator script
├── interpretation/ # Human-readable cluster summaries
├── utils/ # Small, shared helper modules
├── README.md
└── .gitignore


---

## What This Repo *Does Not* Include

- No raw or scraped data
- No user identifiers
- No fine-tuned models for individual profiling
- No externally loaded corpora
- No moral judgments

This is a **mirror**, not a microscope. If you want surveillance, look elsewhere.

---

## License

[MIT License](https://choosealicense.com/licenses/mit/)  
Because open source should not mean open surveillance.

---

## Final Note

This codebase is a mirror:  
Hold it up to a corpus, and it reflects structure.  
Hold it up to ideology, and it reflects contradiction.  
Hold it up to yourself...

For collaborations, philosophical duels, or invitations to build something strange and true: open an issue, or contact me directly.

