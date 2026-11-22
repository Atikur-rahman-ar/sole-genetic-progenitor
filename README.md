# Sole Genetic Progenitors: The Endowed-Pair Model of Single-Couple Human Origin

Author: Atikur Rahman (Independent Researcher, Dhaka, Bangladesh)

This repository is the complete reproducibility package for the paper. It is
fully self-contained: the paper derives the sampled-couple contrast class
inside itself (Section 3.1) and cites no companion or predecessor
manuscript. The paper proves a feasibility theorem for the strongest claim
of the single-couple hypothesis: a founding couple that is the *source* of
the diversity, with no pre-bottleneck population at all; what the theorem
establishes is underdetermination under exact bounds, not actuality. The
couple's genomes enter the model as an initial condition - the
**endowment** h (per-site divergence among the couple's four gene copies),
bounded by the heterozygosity of natural primate genomes (human 3.1-4.1M
SNVs, chimpanzee 5.7M SNVs; Prado-Martinez et al. 2013; Auton et al. 2015).

Archaic admixture is handled by the **descendant-clade extension (DCE)**:
the Neanderthal and Denisovan lineages branch from the couple's descendant
population, so the introgressed 2-6% of non-African genomes is
couple-descended DNA returning through the archaic relatives. The archaic
branches' low heterozygosity is *predicted* (small branches lose the
endowment and carry de-novo diversity only), matching the observed archaic
genomes and independently consistent with the neandersovan bottleneck
(Rogers et al. 2020) and the severe-bottleneck inference of Hu et al. 2023.
The archaic clades' uniparental compartments are the one scoped exception
(clock-free depth-ratio tests), assignable to the published replacement
mechanisms (Y: Mendez et al. 2016; mtDNA: Posth et al. 2017).

All references have been audited; the audit trail is in `citation_audit/`
(summary, full audit, corrections, removed, added - CSV).

## Headline results (all emitted by the fixed-seed script)

- Master equation: E[T2] = D_below + e^(-Lambda_c) [T0 + 0.75 h/(2 mu)]
- Canonical point: T0 = 550 kya, gamma = 1.6, Ne_max = 3e4,
  h = 1.576e-3 (4.73M SNVs/founder) -> E[T2] = 41,671 gen,
  pi = 1.000e-3 (observed 1.0e-3) - exact to 0.01%.
- Joint feasibility window: T0 in [400, 595] kya, where the diversity
  constraint meets the TMR4A compression identity
  (TMR4A = T0 = 495 +/- 100 kya, Swamidass direct-clock estimate),
  coinciding with the independent backward-simulation feasibility of
  Hossjer & Gauger (2019).
- Descendant-clade window: T0 in [550, 595] kya - the archaic clades
  branch at founding; full nuclear coverage holds at the canonical point.
- Uniparental MRCAs = the couple (forced); published mtDNA/Y dates match
  the window under the published ancestral-rate correction s = 2
  (Scally & Durbin 2012) - a conditional, corroborative layer; the window
  itself rests on autosomal evidence alone.
- SFS/Tajima's D: pi_total = 1.001e-3, D = -0.93 (observed ~-2 in
  large-sample series, Tennessen et al. 2012; the residual is addressed
  qualitatively, not by simulation - Section 15).
- LD: E[r^2] = 0.40 at 0.1 kb, 0.30 at 1 kb (observed range 0.3-0.5,
  Tenesa et al. 2007; no overshoot).
- Bottleneck invisible to PSMC-resolution reconstruction (implied Ne
  ~1.2e3 in the T0 bin, not N=2).
- HLA: 2.1-2.5 of 4 founder lineages retained; 8,705 candidate new
  allelic lineages in 550 kyr vs ~10^2-10^3 observed; the deep-root
  lineage count (per-locus, conversion-corrected) is the open test.
- Scoping: S1 (non-introgressed genome, >=94-98% of nuclear DNA)
  unconditional; S2 (descendant-clade full nuclear coverage) at
  T0 in [550, 595] kya; S3 (archaic uniparentals included) falsified
  clock-free by the ~2-3x mtDNA depth ratio and the 2.1x (CI 1.7-2.9)
  Y divergence ratio.

## Citation audit (`citation_audit/`)

52 references; 33 verified unchanged, 8 corrected (wrong journals/titles/
volume fixed: Ewing 2008, Hoelzel 2024, Jackson 2022, Mendez 2016,
Szucs 2017, Takahata 1990, von Salome 2007, Prado-Martinez 2013), 6
removed (2 self-citations, 4 non-academic sources incl. the Buggs slides),
10 added (all verified, incl. Rogers 2020, Hu 2023, Posth 2017,
Hossjer & Gauger 2019, Tennessen 2012). No fabricated citations remain.

## Files

- `sole_progenitor_model.py` - the complete reproducibility suite
  (sections A-N; fixed seeds; ~3 min single core; `--quick` ~20 s)
- `outputs/` - all script output tables: `feasibility_grid.csv` (540
  rows), `clock_scan.csv`, `sfs_endowed_pair.csv`,
  `ld_decay_endowed_pair.csv`, `ne_profile_endowed_pair.csv`,
  `hla_regeneration.csv`, `scoping_tests.csv`, `empirical_anchors.csv`,
  `archaic_branch.csv`, `feasibility_summary.json`,
  `feasibility_report.txt`, plus the eight figures `fig1`-`fig8` as
  vector SVG renditions of the same fixed-seed run (the script itself
  emits each figure in PDF and PNG, which is the format the LaTeX
  source references; the SVG copies are checked in for browser viewing)
- `manuscript/` - LaTeX source of the paper (main.tex + part1..part6.tex)
- `citation_audit/` - full citation audit trail (5 CSV sheets)
- `requirements.txt`

## Reproduction

    pip install -r requirements.txt
    python3 sole_progenitor_model.py            # full suite (all tables + figures in PDF/PNG)
    tectonic manuscript/main.tex                # compile the paper (run the script first: LaTeX reads ../outputs/fig*.pdf)
    python3 sole_progenitor_model.py --quick    # fast check

Every number in the paper is emitted by the script with fixed seeds
(coalescent: 20260101+; backbone: 99001; linkage disequilibrium: 88001;
balanced-locus: 66001; coalescence-rate profile: 770001+), so all results
are bit-reproducible. The PDF is compiled from the LaTeX source with
`tectonic manuscript/main.tex`.
