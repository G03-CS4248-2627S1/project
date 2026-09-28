# literature_review.md — prior work on the four core components

Literature search run on 2026-09-28. Its conclusions went into plan v0.5 (§4.3, §4.4, §4.6 and the §5 novelty claims). It covers the verifier, the joint span head, expected-F1 (MBR) decoding and F1-weighted targets. For each component: the closest prior work, what it did, and how our version differs.

- **How carefully each paper was checked** is marked on every entry:
  - **(read)**: the method and results sections were read.
  - **(abstract)**: only the abstract or a listing page was read. Read the paper before relying on details.
  - **(plan ref)**: already cited in plan §15 and not re-checked here.
- The searches were web searches, not a systematic review. "Nothing found" means nothing turned up in a few targeted searches. Before claiming something is new in the paper, check Google Scholar or Semantic Scholar.
- Plan sections are cited as "plan §x" and spec sections as "spec §x".

---

## 1. Summary

| Component | Closest prior work | Is ours new? | Suggested framing |
|---|---|---|---|
| Joint span head | Fajcik et al. (2021): joint span objective on SQuAD v1.1 and Adversarial SQuAD, with the cross-span failure named | **No.** The head, the cross-span problem and the robustness claim are all published | A known component, used because MBR decoding and the verifier need a coherent span distribution. Cite Fajcik as the source of the diagnosis |
| F1-weighted targets | Boundary smoothing (Zhu & Li, 2022), span-level smoothing in NER. F1 Smoothing (Yin et al., 2023), F1-based but per start and end position | **Partly.** Span-level soft targets exist, and F1-based targets exist, but we found nothing that does both for extractive QA | "Span-level F1-weighted targets", positioned between these two |
| Expected-F1 (MBR) decoding | MBR in translation and generation (Kumar & Byrne, 2004; Bertsch et al., 2023). Nothing found for extractive QA | **Probably.** The cleanest novelty claim, pending a Scholar check | Applying MBR to span extraction and showing it depends on a joint span distribution |
| Verifier | RECONSIDER (Iyer et al., 2021): marked-span cross-encoder re-ranker with a listwise softmax over the top K | **No** for the core design. Our training choices are new | A single-passage (SQuAD) version of RECONSIDER, plus F1-weighted listwise targets, sibling and boundary negatives, and a study of where training candidates come from |

**The main consequence for the paper:** the contribution is how the four parts fit together, plus the analysis (E1–E3). No single component is new on its own. This matches the fallback framing in plan §5.

---

## 2. Joint span head

**[Fajcik, Jon & Smrz (2021). Rethinking the Objectives of Extractive Question Answering. MRQA workshop.](https://aclanthology.org/2021.mrqa-1.2/) (read)**

- **Scope:** BIDAF, BERT-base and ALBERT-xxlarge on 6 datasets, including SQuAD v1.1 and Adversarial SQuAD. Maximum span length 30 tokens. 10 runs per setting and paired t-tests.
- **Objectives compared:**
  - independent: P(s)·P(e);
  - conditional: P(s)·P(e|s), as in XLNet, ALBERT and ELECTRA;
  - joint: one softmax over all (s, e) pairs, scored by a similarity function of start and end vectors;
  - compound: the joint objective plus the independent one as an auxiliary loss.
- **Findings:**
  - Plain dot product works best for BERT-type models.
  - The compound objective is best or tied in most settings. The joint objective alone is sometimes worse.
  - Gains are mostly in EM. F1 sometimes drops, e.g. BERT-base on SQuAD v1.1: independent 81.31 EM / 88.65 F1, compound 81.83 / 88.52.
  - +1.28 EM on Adversarial SQuAD for BERT-base.
  - The conditional objective is equal or worse, with overconfident start predictions.
- **They name the cross-span error.** In their Figure 1, BERT gives the largest probability to a span that runs from the start of "little boy" to the end of "fat man". In a manual check of 200 cases where the two models disagreed on SQuAD v1.1, 5% were the independent model choosing a span that covers several possible answers, which earns partial F1. They conclude that independent models "overfit to F1" this way.
- **They suggest the diverse candidates help re-ranking,** citing RECONSIDER. That is the link our plan builds on.
- **What this means for us:**
  - Plan §4.3's head is a variant of their joint objective: bilinear plus separate start and end terms, W starting at zero, and a length bonus.
  - E3 has a qualitative precedent. Our version measures cross-span mass by confidence level and links it to MBR decoding.
  - **Worth considering:** add their compound objective (an auxiliary independent loss) as an ablation or as the default, since they found the joint objective alone less stable.
  - Their "overfit to F1" finding matches the plan §4.3 argument: under an independent model, MBR decoding prefers cross spans.
  - Code: <https://github.com/KNOT-FIT-BUT/JointSpanExtraction>

**[Lee et al. (2016). Learning Recurrent Span Representations for Extractive Question Answering (RaSoR).](https://arxiv.org/abs/1611.01436) (abstract)**
- An early globally normalized model with an explicit representation of every span (BiLSTM era). Fajcik et al. report RaSoR found the joint objective best among those it tried. Cite it as the origin of span enumeration.

**Dozat & Manning (2017), biaffine scoring (plan ref).** The source of the bilinear score form.

---

## 3. F1-weighted targets

**[Zhu & Li (2022). Boundary Smoothing for Named Entity Recognition. ACL.](https://aclanthology.org/2022.acl-long.490/) (abstract)**
- **What it does:** span-based NER that moves probability from each annotated span to nearby spans, as a form of label smoothing.
- **Reported effects:** less overconfidence, better calibration, flatter minima, and results equal to or better than the previous best on 8 NER benchmarks.
- **Why it matters to us:** the closest span-level soft target. It smooths by boundary distance, while we weight by token F1 with the gold answer, and it is NER, not QA. Its calibration result supports our plan to calibrate and then decode by expected F1.

**[Yin et al. (2023). Rethinking Label Smoothing on Multi-hop Question Answering. CCL.](https://arxiv.org/abs/2212.09512) (read, method section)**
- **"F1 Smoothing":**
  - The start target at position t is the sum of F1(span, gold) over all spans that start at t, passed through a softmax over positions. The end target is built the same way.
  - Also a curriculum that decays the smoothing over training (LDLA).
- **Scope:** HotpotQA only.
- **How it differs from ours:** its targets are separate per start and end position and suit an independent head. Ours (plan §4.4) is one target over whole spans, limited to spans that overlap the gold answer, with a temperature and mixed with the one-hot target. A per-position target puts weight on start and end positions that could then combine into a cross span.

**[Zhao et al. (2020). Robust Machine Reading Comprehension by Learning Soft Labels. COLING.](https://aclanthology.org/2020.coling-main.248/) (abstract)**
- **Methods:** three kinds of soft label: label smoothing, word-overlap labels, and a predicted label distribution.
- **Scope:** ALBERT on SQuAD 2.0, NewsQA and Quoref.
- **Why cite it:** the earliest soft-label work for machine reading comprehension that we found. Yin et al. build on it.

**[Xiong, Zhong & Socher (2018). DCN+: Mixed Objective and Deep Residual Coattention for Question Answering. ICLR.](https://arxiv.org/abs/1711.00106) (abstract)**
- **What it does:** adds a policy-gradient loss whose reward is word-overlap (F1) to the cross-entropy loss.
- **Why cite it:** it names the same mismatch we address, where cross-entropy punishes near-miss answers that F1 would partly credit. It uses reinforcement learning where we use soft targets.

**Norouzi et al. (2016), RAML (plan ref).** The general principle behind plan §4.4: a target distribution based on the task reward.

---

## 4. Expected-F1 (MBR) decoding

**Nothing found for extractive QA.** Several searches ("minimum Bayes risk" with extractive QA, SQuAD or span) returned MBR work on translation, speech recognition, summarization, grammatical error correction and translation-error span detection, but nothing for picking answer spans in extractive QA.
- **To do before claiming novelty:** a Google Scholar or Semantic Scholar search. Also check the SQuAD-era ensembling papers, which may have chosen answers by F1 agreement between models without calling it MBR.

**[Bertsch et al. (2023). It's MBR All the Way Down: Modern Generation Techniques Through the Lens of Minimum Bayes Risk. Big Picture Workshop.](https://aclanthology.org/2023.bigpicture-1.9/) (abstract)**
- **What it does:** a survey that shows many recent methods (e.g. self-consistency) are special cases of MBR.
- **Why cite it:** the general background for MBR decoding. It separates the evidence set, which should be large and diverse, from the hypothesis set, which should be high quality. That applies to our choice of k (plan §4.5).

**Kumar & Byrne (2004) (plan ref).** The original MBR decoding paper.

**Related evidence:** Fajcik et al.'s "overfit to F1" finding (§2 above) shows what MBR-like behaviour looks like under an independent model. Our argument is that MBR needs a joint span distribution to avoid choosing cross spans.

---

## 5. Verifier

**[Iyer, Min, Mehdad & Yih (2021). RECONSIDER: Improved Re-Ranking using Span-Focused Cross-Attention for Open Domain Question Answering. NAACL.](https://aclanthology.org/2021.naacl-main.100/) (read, method and results)**
- **Design:**
  - The candidate span is marked in the passage with `[A]` and `[/A]`.
  - The question and marked passage go through BERT, and a learned vector on `[CLS]` gives the score.
  - A softmax over the K candidates gives the ranking.
  - Training uses 1 positive (an exact gold match) and M−1 random negatives per question.
- **Results:**
  - Open-domain: DPR retriever plus BERT reader. State of the art for extractive systems on NQ, TriviaQA, WebQuestions and TREC; the large model improved on DPR by about 1 point.
  - **Span marking is essential:** replacing it with appending the answer text costs about 1% on NQ.
  - K = 5 at test time was best, even though training used the top 100.
- **Training candidates come from the reader's own predictions** on its own training questions: "We obtain top K predictions (pj, sj) of model M for each question qi in its training set". There is no cross-fitting. This is the precedent for plan v0.5's default of sampling the verifier's training candidates from the final reader (plan §4.6).
- **How ours differs:**
  - Single passage (SQuAD) instead of open-domain.
  - Soft F1-weighted listwise targets instead of one exact positive.
  - Sibling-answer and ±1-word boundary negatives.
  - The verifier score is combined with the reader's log-probability and then calibrated.
  - An explicit comparison of where training candidates come from (E2).
- Code: <https://github.com/facebookresearch/reconsider>

**[Charniak & Johnson (2005). Coarse-to-Fine n-Best Parsing and MaxEnt Discriminative Reranking. ACL.](https://aclanthology.org/P05-1022/) (read, relevant sections)**
- **Cross-fitting:** they used 20-fold cross-validation (following Collins, 2000) to produce the reranker's training candidates, "to avoid training the n-best parser on the sentence it was being used to parse". This is the classic precedent for plan §4.6's cross-fitting.
- **Metric-based target:** the gold parse is often not in the candidate list, so the reranker is trained to pick the candidate with the best F-score. This is like our F1-weighted listwise targets when the gold answer is missing from the top-k.
- **For plan §4.6:** together with RECONSIDER, this gives one strong precedent for each side of the in-sample vs. cross-fitted comparison in E2.

**[Wang et al. (2018). Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering. ICLR.](https://arxiv.org/abs/1711.05116) (abstract)**
- Re-ranks a QA model's candidate answers by pooling evidence across passages ("strength" and "coverage"). It is open-domain and multi-passage, so it is less close to ours. Cite it as background on answer re-ranking.

**Hu et al. (2019), Read+Verify, and Zhang et al. (2021), Retro-Reader (plan ref).** Both are SQuAD 2.0 verifiers that mainly decide whether a question can be answered at all, which doesn't apply to v1.1. Cite them for the name "verifier" and the idea of reading twice.

---

## 6. Suggested follow-ups

1. **Done in plan v0.5:** the §5 novelty claims were rewritten along these lines, prior work was added to §4.3, §4.4 and §4.6, and the verifier now trains on in-sample (sampled) candidates, with cross-fitting as the E2 ablation.
2. **Joint head ablation (plan §6.2):** consider adding Fajcik et al.'s compound objective (joint + auxiliary independent loss). Not yet in the plan.
3. **In-sample candidates:** RECONSIDER shows they can work, but its reader was trained on distantly supervised open-domain data, so it probably fit its training questions less closely than a SQuAD reader will. That is why plan §4.6 samples at a temperature above 1 and checks the candidates against real validation mistakes.
4. **Before writing related work:** read the four abstract-only entries in full, and run the Scholar check for MBR in extractive QA.
