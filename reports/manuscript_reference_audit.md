# Manuscript reference-preservation audit

## Baseline and result

The author-designated reference baseline was `manuscript/Heatwave_Revised.docx` in the original working tree (SHA-256 `6b57b3810d55492f6e70acfe0eba9f2bbaf3df152b4c0ae05d734b80ae3454bd`). It contained 54 nonempty bibliography entries between the `References` and `Statements and Declarations` headings.

All 54 baseline entries are present in `manuscript/original_article_clean.md`. No baseline reference was deleted. Seven entries that had disappeared during an earlier Markdown rewrite—Breiman (2001), Chen and Guestrin (2016), Dickey and Fuller (1979), Global Forest Watch (2024), Hochreiter and Schmidhuber (1997), Meteoblue (2024), and Rothfusz (1990)—were restored.

The integrated manuscript contains 66 references: the original 54 plus 12 additions. The additions are Chaki et al. (2025), Choudary V et al. (2025), Farukh et al. (2026), Islam et al. (2024), Kan et al. (2025), Khatun et al. (2017), Lundberg and Lee (2017), Mallik et al. (2024), Molla et al. (2025), Rashid et al. (2024), Saito and Rehmsmeier (2015), and Van Calster et al. (2019).

## Why the additions were retained

- Choudary V et al. and Kan et al. are the two papers whose transferable prediction methods motivated the benchmark.
- Chaki et al., Farukh et al., Islam et al., Khatun et al., Mallik et al., Molla et al., and Rashid et al. strengthen the Bangladesh, Dhaka, and regional positioning.
- Saito and Rehmsmeier support prevalence-aware precision–recall evaluation.
- Van Calster et al. support explicit calibration assessment.
- Lundberg and Lee support the SHAP method and its restricted predictive interpretation.

The bibliography is intentionally larger than the prior 54-entry list because preservation and analytical relevance were prioritized over maintaining an arbitrary reference count. References attached only to legacy analyses are cited in the supplementary section that explains why those analyses were not retained as article evidence.
