"""sole_progenitor_model.py -- Endowed-Pair Model for Single-Couple Human Origin.

Full reproducibility suite for the manuscript

    "Sole Genetic Progenitors: The Endowed-Pair Model of Single-Couple
     Human Origin".

The Endowed-Pair (EP) model has no pre-bottleneck population: the founding
couple (N=2) is not a random sample drawn from a gene pool of size N_pre,
but carries an *internal endowment* -- its four gene copies per locus carry
up to four alleles whose mutual divergence h is bounded by the heterozygosity
observed in natural primate genomes (human 3.1-4.1M SNVs, chimpanzee 5.7M
SNVs, Prado-Martinez et al. 2013; Auton et al. 2015). Archaic admixture is
treated within the descendant-clade extension (Section K/N): the
Neanderthal/Denisovan clades branch from the couple's descendants, so
introgressed tracts are couple-descended DNA returning through the archaic
branches; descendant branches that stayed small carry heterozygosities
several-fold below the modern branch, as observed.

Everything the manuscript reports is emitted by this script with fixed seeds:

  Section A  calibration constants and endowment capacity bounds
  Section B  transmission probabilities of a four-allele endowment
  Section C  master coalescent equation of the EP model (E[T2], pi)
  Section D  feasibility grid over (T0, Ne_max, gamma, h) -> feasible window
  Section E  TMR4A compression identity and joint feasibility window
  Section F  uniparental clock scan (mtDNA/Y MRCA under ancestral-rate correction)
  Section G  site-frequency spectrum + Tajima's D (msprime de-novo + endowment backbone)
  Section H  linkage-disequilibrium decay r^2(d) under the EP model
  Section I  coalescence-rate profile / PSMC-style Ne reconstruction (bottleneck invisibility)
  Section J  balanced-locus (HLA-class) retention and regeneration
  Section K  scoping tests: rate-free depth-ratio tests and claim scoping
  Section L  empirical existence-proof anchors
  Section M  summary outputs (JSON + report) and figures
  Section N  descendant-clade extension: archaic-branch heterozygosity

Run:  python3 sole_progenitor_model.py            (full suite, ~3 min)
      python3 sole_progenitor_model.py --quick    (reduced replicates, ~20 s)

All times in generations unless marked kya (1 generation = 25 years).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# Global calibration
# ----------------------------------------------------------------------------
MU = 1.2e-8            # per-site per-generation mutation rate (Kong 2012, Jonsson 2017)
RECOMB = 1.0e-8        # per-site per-generation recombination rate
GEN_YEARS = 25.0       # years per generation
L_FULL = 3.0e9         # genome size (bp) used for SNV-count conversions
PI_OBS = 1.0e-3        # observed genome-wide nucleotide diversity (Auton 2015)
E_T2_REQ = PI_OBS / (2.0 * MU)          # = 41,667 generations
T_RECENT = 2000        # recent-growth epoch length (generations)
NE_RECENT = 1.0e6      # present-day SFS-scale Ne under recent growth

TMR4A_EST = 495.0      # kya, Swamidass TMR4A estimate (Swamidass 2019 book;
                       # 2022 conference paper: median TMR4A ~ 500 kya)
TMR4A_CI = 100.0       # +/- kya

MTDNA_EST_LO, MTDNA_EST_HI = 150.0, 200.0   # kya, canonical mtDNA-Eve point range
MTDNA_CI_LO, MTDNA_CI_HI = 129.0, 298.0     # kya, Vigilant 1991 / Cann 1987 CIs
Y_EST_LO, Y_EST_HI = 120.0, 338.0           # kya, Poznik 2013 ... Mendez 2013 (A00) span
Y_CI_LO, Y_CI_HI = 115.0, 338.0             # kya

OUT = None              # set by main()
QUICK = False


def kya_to_gens(kya):
    return kya * 1e3 / GEN_YEARS


def gens_to_kya(g):
    return g * GEN_YEARS / 1e3


# ----------------------------------------------------------------------------
# Section A. Endowment capacity of two genomes
# ----------------------------------------------------------------------------
def section_a():
    """Biological bounds on the endowment parameter h (per-site divergence
    between the founding couple's gene copies).

    h is bounded above by the heterozygosity carried by natural primate
    genomes: the chimpanzee carries ~5.7M SNVs and the modern human
    3.1-4.1M (Prado-Martinez et al. 2013; Auton et al. 2015). Two founders
    whose SNV sets are largely disjoint carry a pool heterozygosity of
    ~0.75*h (each single-founder-heterozygous site contributes
    2*(1/4)*(3/4) = 0.375 per founder).
    """
    snv = {
        "human_avg": 4.08e6,     # Auton et al. 2015 (avg sample SNVs)
        "human_min": 3.1e6,      # Prado-Martinez et al. 2013
        "chimp": 5.7e6,          # Prado-Martinez et al. 2013
    }
    rows = []
    for k, m in snv.items():
        rows.append((k, m, m / L_FULL, 0.75 * m / L_FULL))
    print("== Section A: endowment capacity ====================================")
    print(f"{'genome':<12}{'SNVs':>12}{'h (per-site)':>15}{'pool H (0.75h)':>17}")
    for k, m, h, hp in rows:
        print(f"{k:<12}{m:>12,.0f}{h:>15.2e}{hp:>17.2e}")
    # the canonical endowment used in the manuscript
    h_human = snv["human_avg"] / L_FULL
    h_chimp = snv["chimp"] / L_FULL
    h_canon = 1.576e-3      # ~4.73M SNVs per founder: between human and chimp scale;
                            # yields pi = 1.000e-3 at the canonical point (550 kya)
    print(f"\nCanonical endowment h = {h_canon:.2e} "
          f"({h_canon*L_FULL/1e6:.2f}M SNVs/founder = "
          f"{h_canon/h_chimp*100:.0f}% of chimp, {h_canon/h_human*100:.0f}% of 1000G human avg)")
    print(f"Master-equation depth of the canonical endowment: "
          f"0.75*h/(2*mu) = {0.75*h_canon/(2*MU):,.0f} generations "
          f"= {gens_to_kya(0.75*h_canon/(2*MU)):.2f} kya of coalescent-equivalent depth")
    return dict(snv=snv, h_human=h_human, h_chimp=h_chimp, h_canon=h_canon)


# ----------------------------------------------------------------------------
# Section B. Transmission of a four-allele endowment through expansion
# ----------------------------------------------------------------------------
def section_b():
    """Probability that all four founder alleles at a locus are transmitted.

    Father carries alleles {A,B}, mother {C,D}; each child receives one
    paternal and one maternal allele independently. The events "both paternal
    alleles appear among n children" and "both maternal alleles appear" are
    independent, each with probability 1 - 2*(1/2)^n, so

        P(all four alleles transmitted | n children) = [1 - 2^(1-n)]^2 .

    The formula gives 0.25 (n=2), 0.56 (n=3), 0.77 (n=4), saturating to
    0.996 by n=10.
    """
    print("\n== Section B: four-allele transmission ==============================")
    rows = []
    for n in range(2, 13):
        p_both = 1.0 - 2.0 * 0.5 ** n
        p4 = p_both ** 2
        rows.append((n, p_both, p4))
        print(f"n={n:>2} children: P(both paternal)={p_both:.3f}  "
              f"P(all four alleles)={p4:.3f}")
    # genome-wide retention through the growth trajectory: per-locus loss is
    # governed by the drift hazard; the expected retained fraction of the
    # endowment equals exp(-Lambda) with Lambda from Section C.
    return rows


# ----------------------------------------------------------------------------
# Section C. Master coalescent equation of the Endowed-Pair model
# ----------------------------------------------------------------------------
def growth_span(gamma, ne_max):
    return int(np.ceil(np.log(ne_max / 2.0) / np.log(gamma)))


def backward_sizes(T0, gamma, ne_max, include_recent=True):
    """Backward-time N(t) for t = 0 .. T0-1 (before the couple generation).

    t in [0, T_RECENT): recent-growth epoch (forward ne_max -> NE_RECENT)
    t in [T_RECENT, T0-G): plateau at ne_max
    t in [T0-G, T0-1): fast expansion, N = 2*gamma^(T0-1-t)
    (the couple generation itself, forward g=0, N=2, is handled separately)
    """
    G = growth_span(gamma, ne_max)
    G = min(G, T0 - 1)
    r = np.log(NE_RECENT / ne_max) / T_RECENT
    ts = np.arange(T0 - 1)                      # backward times 0 .. T0-2
    n = np.empty(len(ts))
    for i, t in enumerate(ts):
        if include_recent and t < T_RECENT:
            g_fwd = T0 - 1 - t
            val = ne_max * np.exp(r * (g_fwd - (T0 - T_RECENT)))
            n[i] = min(val, NE_RECENT)
        elif t < T0 - G:
            n[i] = ne_max
        else:
            g_fwd = T0 - 1 - t
            n[i] = min(2.0 * gamma ** g_fwd, ne_max)
    return n


def master_equation(T0, gamma, ne_max, h_endow, include_recent=True):
    """Endowed-pair master equation.

    Backward from the present, a pair of gene copies coalesces during the
    post-bottleneck history (plateau + expansion) with the standard hazard
    lambda(t) = 1/(2 N(t)); the density of coalescence at backward time t is
    lambda(t) * exp(-Lambda(t)). Pairs that survive to the couple generation
    (probability e^{-Lambda_c}, where Lambda_c is the hazard accumulated up
    to the couple generation) pass through the four founder copies:

      * with probability 1/4 the pair lands on the same founder copy and
        coalesces at T0;
      * with probability 3/4 the pair lands on two different founder copies.
        Those copies are unrelated *within* the model: their divergence is
        the endowment itself, h, equivalent to extra coalescent depth
        h/(2*mu). (For two founders with disjoint SNV sets and comparable
        heterozygosity h, the within-founder and cross-founder divergences
        are both ~ h, so the mixture collapses to a single depth.)

    Hence

      E[T2] = D_<c + e^{-Lambda_c} * [ T0 + (3/4) * h/(2 mu) ],

    where D_<c = integral t lambda(t) e^{-Lambda(t)} dt is the expected depth
    contributed by pairs coalescing before the couple, and

      pi = 2 mu E[T2].
    """
    n = backward_sizes(T0, gamma, ne_max, include_recent)
    lam = 1.0 / (2.0 * n)
    Lam = np.cumsum(lam)                        # Lambda(t) up to and including t
    Lam_c = Lam[-1]                             # hazard to couple generation
    D_below = float(np.sum(lam * np.exp(-Lam) * np.arange(len(lam))))
    depth_endow = 0.75 * h_endow / (2.0 * MU)
    E_T2 = D_below + np.exp(-Lam_c) * (T0 + depth_endow)
    return dict(E_T2=E_T2, pi=2.0 * MU * E_T2, Lambda_c=Lam_c,
                D_below=D_below, eLam=np.exp(-Lam_c), depth_endow=depth_endow,
                n=n, lam=lam, Lam=Lam)


# ----------------------------------------------------------------------------
# Section D. Feasibility grid
# ----------------------------------------------------------------------------
def section_d(h_canon):
    print("\n== Section D: feasibility grid ======================================")
    T0_kya_list = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800]
    ne_list = [1e4, 2e4, 3e4, 5e4, 1e5]
    gam_list = [1.3, 1.6, 2.0]
    h_list = [1.37e-3, 1.576e-3, 1.90e-3]  # includes the canonical endowment
    rows = []
    for t0k in T0_kya_list:
        T0 = int(round(kya_to_gens(t0k)))
        for ne in ne_list:
            for gam in gam_list:
                for h in h_list:
                    me = master_equation(T0, gam, ne, h)
                    ok = abs(me["pi"] - PI_OBS) <= 0.15 * PI_OBS
                    rows.append(dict(T0_kya=t0k, T0_gens=T0, ne_max=ne, gamma=gam,
                                     h=h, E_T2=me["E_T2"], pi=me["pi"],
                                     Lambda_c=me["Lambda_c"], eLam=me["eLam"],
                                     D_below=me["D_below"], feasible=ok))
    # write CSV
    keys = list(rows[0].keys())
    with open(f"{OUT}/feasibility_grid.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"  grid points: {len(rows)}  feasible (|pi-1e-3| <= 15%): "
          f"{sum(r['feasible'] for r in rows)}")
    # canonical point
    T0c = int(round(kya_to_gens(550.0)))
    me = master_equation(T0c, 1.6, 3e4, h_canon)
    print(f"  canonical (T0=550 kya, gamma=1.6, Ne_max=3e4, h={h_canon:.2e}): "
          f"E[T2]={me['E_T2']:,.0f} gen, pi={me['pi']*1e3:.3f}e-3, "
          f"Lambda_c={me['Lambda_c']:.3f}, D_below={me['D_below']:,.0f}")
    # feasible T0 window per Ne_max at canonical gamma and h
    print("\n  Feasible T0 window at gamma=1.6, h=h_canon (kya):")
    win = {}
    for ne in ne_list:
        okk = [r["T0_kya"] for r in rows if r["ne_max"] == ne and r["gamma"] == 1.6
               and abs(r["h"] - h_canon) < 1e-12 and r["feasible"]]
        win[ne] = (min(okk), max(okk)) if okk else None
        print(f"    Ne_max={ne:>8.0f}: {win[ne]}")
    return rows, win


# ----------------------------------------------------------------------------
# Section E. TMR4A compression identity and the joint window
# ----------------------------------------------------------------------------
def section_e(rows, h_canon):
    print("\n== Section E: TMR4A identity and joint window =======================")
    # Under the EP model every locus passes through the couple's four gene
    # copies: the genome-wide four-lineage compression time (TMR4A) equals T0
    # exactly. Swamidass's direct-clock estimate is 495 +/- 100 kya.
    lo, hi = TMR4A_EST - TMR4A_CI, TMR4A_EST + TMR4A_CI
    print(f"  TMR4A_true = T0 (compression identity)")
    print(f"  direct-clock estimate: {TMR4A_EST} +/- {TMR4A_CI} kya -> window [{lo}, {hi}] kya")
    # joint window: T0 feasible for pi AND inside TMR4A CI
    feas = sorted(set(r["T0_kya"] for r in rows if r["feasible"]))
    joint = [t for t in feas if lo <= t <= hi]
    print(f"  pi-feasible T0 grid values: {feas}")
    print(f"  JOINT window (pi-feasible AND TMR4A-CI): {joint}")
    print(f"  nominal T0 = 550 kya (upper window: descendant-clade coverage available)")
    # observed autosomal coalescence depth check
    me = master_equation(int(round(kya_to_gens(550.0))), 1.6, 3e4, h_canon)
    print(f"  expected pairwise depth {gens_to_kya(me['E_T2']):.2f} kya "
          f"(observed autosomal range 500-2000 kya: "
          f"{'inside' if 500 <= gens_to_kya(me['E_T2']) <= 2000 else 'CHECK'})")
    return joint


# ----------------------------------------------------------------------------
# Section F. Uniparental clock scan
# ----------------------------------------------------------------------------
def section_f(joint_window):
    print("\n== Section F: uniparental clock scan ================================")
    """mtDNA and Y MRCAs are forced to T0 under sole progenitorship. Their
    published date estimates are anchored on the human-chimp split, a clock
    that Scally & Durbin (2012) showed requires an ancestral-rate correction
    of about a factor 2 relative to direct trio-measured rates. The scan
    multiplies the published mtDNA/Y estimates by s in [1.0, 2.5] and checks
    containment of the joint window."""
    rows = []
    for s in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]:
        mt_lo, mt_hi = MTDNA_CI_LO * s, MTDNA_CI_HI * s
        y_lo, y_hi = Y_CI_LO * s, Y_CI_HI * s
        mt_pt_lo, mt_pt_hi = MTDNA_EST_LO * s, MTDNA_EST_HI * s
        y_pt_lo, y_pt_hi = Y_EST_LO * s, Y_EST_HI * s
        contains = mt_lo <= joint_window[0] and joint_window[-1] <= mt_hi
        rows.append(dict(s=s, mt_ci=(mt_lo, mt_hi), y_ci=(y_lo, y_hi),
                         mt_point=(mt_pt_lo, mt_pt_hi), y_point=(y_pt_lo, y_pt_hi),
                         contains_joint=contains))
        print(f"  s={s:>4}: mtDNA CI [{mt_lo:.0f},{mt_hi:.0f}] kya, "
              f"Y span [{y_lo:.0f},{y_hi:.0f}] kya -> "
              f"contains joint window: {contains}")
    with open(f"{OUT}/clock_scan.csv", "w") as f:
        f.write("s,mt_ci_lo,mt_ci_hi,y_ci_lo,y_ci_hi,contains_joint\n")
        for r in rows:
            f.write(f"{r['s']},{r['mt_ci'][0]},{r['mt_ci'][1]},{r['y_ci'][0]},"
                    f"{r['y_ci'][1]},{r['contains_joint']}\n")
    return rows


# ----------------------------------------------------------------------------
# Section G. SFS + Tajima's D (msprime de-novo + endowment backbone)
# ----------------------------------------------------------------------------
def backward_size_fn(T0, gamma, ne):
    """Backward-time population size function used by both msprime sections."""
    G = growth_span(gamma, ne)
    r = np.log(NE_RECENT / ne) / T_RECENT

    def sz(t):
        if t < T_RECENT:
            g_fwd = T0 - 1 - t
            return min(ne * np.exp(r * (g_fwd - (T0 - T_RECENT))), NE_RECENT)
        if t < T0 - G:
            return ne
        g_fwd = T0 - 1 - t
        return min(2.0 * gamma ** g_fwd, ne)
    return sz


def msprime_events(T0, gamma, ne, bin_w=200):
    import msprime
    """Piecewise-constant demographic events for msprime.simulate.

    Backward epochs: recent growth [0, T_RECENT), plateau ne, fast expansion
    to N=2 in the last generation before T0, then a small sink above T0
    (mutations there are filtered out -- they correspond to the endowment,
    handled separately)."""
    sz = backward_size_fn(T0, gamma, ne)
    events = []
    for t in range(bin_w, T0, bin_w):
        events.append(msprime.PopulationParametersChange(time=t, initial_size=sz(t)))
    events.append(msprime.PopulationParametersChange(time=T0, initial_size=1e3))
    return events


def section_g(h_canon):
    import msprime
    print("\n== Section G: site-frequency spectrum and Tajima's D ================")
    T0 = int(round(kya_to_gens(550.0)))          # 22,000
    ne = 3e4
    n_rep = 40 if QUICK else 200
    Lsim = 2e6 if QUICK else 5e6                 # simulated bp for de-novo SFS
    n_samp = 100                                 # sample size (haploid gene copies)

    events = msprime_events(T0, 1.6, ne)
    seed0 = 20260101
    fs_denovo = np.zeros(n_samp - 1)
    S_denovo = 0.0
    pi_denovo_num = 0.0
    total_len = 0.0
    for i in range(n_rep):
        ts = msprime.simulate(sample_size=n_samp, Ne=NE_RECENT,
                              length=Lsim, recombination_rate=RECOMB,
                              mutation_rate=MU, demographic_events=events,
                              random_seed=seed0 + i)
        total_len += ts.sequence_length
        for tree in ts.trees():
            for mut in tree.mutations():
                if mut.time >= T0 - 1.0:         # above the couple: endowment zone
                    continue
                c = tree.num_samples(mut.node)
                if 1 <= c <= n_samp - 1:
                    fs_denovo[c - 1] += 1
                    S_denovo += 1
                    pi_denovo_num += 2.0 * c * (n_samp - c) / n_samp ** 2
    # normalize per-site
    fs = fs_denovo / total_len
    pi_denovo = pi_denovo_num / total_len
    thetaW_denovo = S_denovo / total_len / float(np.sum(1.0 / np.arange(1, n_samp)))
    # endowment backbone contribution (analytic + plateau-absorption sim)
    bb = endowment_backbone(T0, 1.6, ne, h_canon, n_samp)
    pi_total = pi_denovo + bb["pi_per_site"]
    S_total_per_site = thetaW_denovo * float(np.sum(1.0 / np.arange(1, n_samp))) + bb["S_per_site"]
    thetaW_total = S_total_per_site / float(np.sum(1.0 / np.arange(1, n_samp)))
    # Tajima's D on a 100-kb window convention
    W = 1e5
    a1 = float(np.sum(1.0 / np.arange(1, n_samp)))
    a2 = float(np.sum(1.0 / np.arange(1, n_samp) ** 2))
    b1 = (n_samp + 1) / (3.0 * (n_samp - 1))
    b2 = 2.0 * (n_samp ** 2 + n_samp + 3) / (9.0 * n_samp * (n_samp - 1))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n_samp + 2) / (a1 * n_samp) + a2 / a1 ** 2
    e1 = c1 / a1
    e2 = c2 / (a1 ** 2 + a2)
    S_window = S_total_per_site * W
    thetaW_window = S_window / a1
    pi_window = pi_total * W                 # convert to per-window count units
    D = (pi_window - thetaW_window) / np.sqrt(e1 * S_window + e2 * S_window * (S_window - 1))
    print(f"  de-novo (msprime, {n_rep} reps x {Lsim:.1e} bp, n={n_samp}): "
          f"pi={pi_denovo*1e3:.3f}e-3, theta_W={thetaW_denovo*1e3:.3f}e-3")
    print(f"  endowment backbone: pi={bb['pi_per_site']*1e3:.3f}e-3, "
          f"S={bb['S_per_site']:.2e}/site")
    print(f"  TOTAL: pi={pi_total*1e3:.3f}e-3 (observed {PI_OBS*1e3:.1f}e-3), "
          f"theta_W={thetaW_total*1e3:.3f}e-3, Tajima's D(100 kb window)={D:.2f} "
          f"(observed approx -2 in large-sample series, Tennessen et al. 2012)")
    # save SFS
    with open(f"{OUT}/sfs_endowed_pair.csv", "w") as f:
        f.write("derived_count,per_site_frequency\n")
        for i, v in enumerate(fs):
            f.write(f"{i+1},{v}\n")
    return dict(fs=fs, pi_denovo=pi_denovo, pi_bb=bb["pi_per_site"], pi_total=pi_total,
                thetaW_total=thetaW_total, D=D, bb=bb)


def endowment_backbone(T0, gamma, ne, h, n_samp, Lam_c=None):
    """Endowment-backbone contribution to present-day diversity.

    Expected heterozygosity decays under drift exactly as

        pi_bb = H_pool(T0) * prod_t (1 - 1/(2N(t))) = 0.75 * h * e^{-Lambda_c},

    which is precisely the endowment term of the master equation (the two
    views -- per-pair depth and per-site heterozygosity -- agree exactly).
    The forward-WF simulation below is used only for the SEGREGATING-SITE
    fraction S_bb: B sites each starting as a four-allele locus (1,1,1,1);
    exact multinomial WF during the fast expansion, variance-matched
    diffusion steps on the focal allele during the plateau and recent-growth
    epochs. Mixed 2/3/4-allele endowment configurations are approximated by
    the four-allele case (documented approximation).
    """
    B = 20000 if QUICK else 120000
    G = growth_span(gamma, ne)
    if Lam_c is None:
        Lam_c = master_equation(T0, gamma, ne, h)["Lambda_c"]
    pi_bb = 0.75 * h * float(np.exp(-Lam_c))
    rng = np.random.default_rng(99001)
    counts = np.ones((B, 4))
    g0 = int(np.log(1000.0) / np.log(gamma))
    for g in range(min(g0, G)):
        N = min(2.0 * gamma ** g, ne)
        p = counts / counts.sum(axis=1, keepdims=True)
        counts = rng.multinomial(max(int(2 * N), 2), p)
    # remaining expansion + plateau + recent growth: focal-allele diffusion
    p_now = counts[:, 0] / counts.sum(axis=1, keepdims=True)[:, 0]
    step = 100
    n_steps = (T0 - g0) // step
    seg = (p_now > 0) & (p_now < 1)
    for _ in range(n_steps):
        var = p_now * (1 - p_now) * step / (2 * ne)
        p_new = np.clip(p_now + rng.normal(0, 1, len(p_now)) * np.sqrt(var), 0.0, 1.0)
        seg = seg & (p_new > 0) & (p_new < 1)
        p_now = np.where(seg, p_new, p_now)
    S_bb = float(np.mean(seg)) * 2.0 * h     # x (2h) = density of endowed sites per bp
    return dict(pi_per_site=pi_bb, S_per_site=S_bb, B=B, G=G, seg_frac=float(np.mean(seg)))


# ----------------------------------------------------------------------------
# Section H. Linkage disequilibrium decay
# ----------------------------------------------------------------------------
def section_h():
    print("\n== Section H: linkage-disequilibrium decay ==========================")
    T0 = int(round(kya_to_gens(550.0)))
    ne = 3e4
    G = growth_span(1.6, ne)
    step = 100
    n_rep = 400 if QUICK else 2000
    rng = np.random.default_rng(88001)
    distances_kb = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    rows = []
    # Initial condition: the couple's four haploid genomes at two loci. Phase is
    # random: with probability 1/2 the founders' alleles are in the same phase
    # (haplotypes AB, ab at 1/2 each: r^2 = 1) and with probability 1/2 in
    # opposite phases (four haplotypes at 1/4: r^2 = 0). E[r0^2] = 1/2.
    half = n_rep // 2
    for dkb in distances_kb:
        c = RECOMB * dkb * 1000.0
        hap = np.empty((n_rep, 4))
        hap[:half] = [0.5, 0.0, 0.0, 0.5]
        hap[half:] = [0.25, 0.25, 0.25, 0.25]
        # fast expansion: exact WF while drift is appreciable (2N <= 2000)
        g0 = int(np.log(1000.0) / np.log(1.6))          # 2*1.6^g <= 2000
        for g in range(g0):
            N = min(2.0 * 1.6 ** g, ne)
            hap = rng.multinomial(max(int(2 * N), 2), hap) / (2.0 * N)
            # recombination each generation
            pA = hap[:, 0] + hap[:, 1]
            pB = hap[:, 0] + hap[:, 2]
            D = hap[:, 0] - pA * pB
            D *= (1.0 - c)
            hap[:, 0] = pA * pB + D
            hap[:, 1] = pA * (1 - pB) - D
            hap[:, 2] = (1 - pA) * pB - D
            hap[:, 3] = (1 - pA) * (1 - pB) + D
            hap = np.clip(hap, 0, None)
            hap = hap / hap.sum(axis=1, keepdims=True)
        # plateau: exact WF drift via time-dilation. m=333 rescaled generations
        # at 2N'=1000 reproduce the drift of ~20,000 generations at 2N=60,000
        # (equal cumulative hazard tau = m/(2N')); recombination is applied per
        # rescaled generation at the dilated rate (1-c)^(T0/m).
        twon_eff = 1000.0
        m_resc = max(int(round(T0 / (2.0 * ne) * twon_eff)), 1)
        c_dil = 1.0 - (1.0 - c) ** (T0 / m_resc)
        for _ in range(m_resc):
            hap = rng.multinomial(int(twon_eff), hap) / twon_eff
            pA = hap[:, 0] + hap[:, 1]
            pB = hap[:, 0] + hap[:, 2]
            D = hap[:, 0] - pA * pB
            D *= (1.0 - c_dil)
            hap[:, 0] = pA * pB + D
            hap[:, 1] = pA * (1 - pB) - D
            hap[:, 2] = (1 - pA) * pB - D
            hap[:, 3] = (1 - pA) * (1 - pB) + D
            hap = np.clip(hap, 0, None)
            hap = hap / hap.sum(axis=1, keepdims=True)
        pA = hap[:, 0] + hap[:, 1]
        pB = hap[:, 0] + hap[:, 2]
        Dm = hap[:, 0] - pA * pB
        denom = pA * (1 - pA) * pB * (1 - pB)
        ok = denom > 1e-12
        r2 = np.zeros_like(Dm)
        r2[ok] = Dm[ok] ** 2 / denom[ok]
        rows.append(dict(d_kb=dkb, c=c, r2_mean=float(np.mean(r2)),
                         r2_median=float(np.median(r2))))
        print(f"  d={dkb:>6.1f} kb (c={c:.1e}): E[r^2]={np.mean(r2):.3f}  "
              f"median={np.median(r2):.3f}")
    with open(f"{OUT}/ld_decay_endowed_pair.csv", "w") as f:
        f.write("d_kb,r2_mean,r2_median\n")
        for r in rows:
            f.write(f"{r['d_kb']},{r['r2_mean']},{r['r2_median']}\n")
    print("  observed short-range r^2 at 1 kb ~ 0.3-0.5 (population- and")
    print("  minor-allele-frequency convention dependent; Tenesa et al. 2007)")
    print("  (backbone pairs = pairs of endowed common variants; genome-wide")
    print("   averages are further diluted by young de-novo pairs, which show")
    print("   lower LD, so the genome-wide profile sits at or below these values)")
    return rows


# ----------------------------------------------------------------------------
# Section I. Coalescence-rate profile (PSMC-style Ne reconstruction)
# ----------------------------------------------------------------------------
def section_i():
    import msprime
    print("\n== Section I: coalescence-rate profile / bottleneck invisibility ====")
    T0 = int(round(kya_to_gens(550.0)))
    ne = 3e4
    events = msprime_events(T0, 1.6, ne)
    n_rep = 40 if QUICK else 200
    seeds = 770001 + np.arange(n_rep)
    tmrcas = []
    L = 2e6 if QUICK else 1e7
    for sd in seeds:
        ts = msprime.simulate(sample_size=2, Ne=NE_RECENT, length=L,
                              recombination_rate=RECOMB, mutation_rate=0,
                              demographic_events=events, random_seed=int(sd))
        t_list = []
        for tree in ts.trees():
            t_list.append(tree.tmrca(0, 1))
        tmrcas.append(np.array(t_list))
    # PSMC-style: 32 time bins matching PSMC's temporal resolution (~15-30 kyr
    # bins at 0.5 Myr depth); invert the pairwise-coalescence hazard per bin
    allT = np.concatenate(tmrcas)
    edges = np.logspace(2.7, np.log10(1.5e5), 33)
    hist, _ = np.histogram(allT, bins=edges)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    surv = 1.0 - (np.cumsum(hist) - hist) / hist.sum()
    surv = np.clip(surv, 1e-8, None)
    with np.errstate(divide="ignore"):
        lam = -np.diff(np.log(surv)) / widths[:-1]
    ne_implied = 1.0 / (2.0 * np.maximum(lam, 1e-12))
    with open(f"{OUT}/ne_profile_endowed_pair.csv", "w") as f:
        f.write("t_gen,t_kya,ne_implied\n")
        for t, nv in zip(centers, ne_implied):
            f.write(f"{t:.0f},{gens_to_kya(t):.1f},{nv:.0f}\n")
    # the dip test: implied Ne in the bin containing T0 (the couple generation)
    idx = int(np.argmin(np.abs(centers - T0)))
    idx = min(idx, len(ne_implied) - 1)
    print(f"  true N(T0) = 2 (the couple); PSMC-resolution implied Ne in the "
          f"T0 bin = {ne_implied[idx]:,.0f}")
    print(f"  -> at PSMC/MSMC temporal resolution the one-generation bottleneck is")
    print(f"     invisible: the inferred profile tracks the plateau and shows no dip")
    print(f"     (resolution limits: Li & Durbin 2011)")
    return dict(centers=centers, ne_implied=ne_implied, idx_t0=idx)


# ----------------------------------------------------------------------------
# Section J. Balanced-locus (HLA-class) retention and regeneration
# ----------------------------------------------------------------------------
def section_j():
    print("\n== Section J: balanced-locus retention and regeneration =============")
    T0 = int(round(kya_to_gens(550.0)))
    ne = 3e4
    G = growth_span(1.6, ne)
    n_rep = 300 if QUICK else 1500
    rng = np.random.default_rng(66001)
    g0 = int(np.log(1000.0) / np.log(1.6))          # exact WF while 2N <= 2000

    def retention_sim(s_over, gamma=1.6):
        surv_counts = []
        for rep in range(n_rep):
            counts = np.array([1, 1, 1, 1], dtype=float)   # 4 founder lineages
            # expansion with overdominant protection: marginal fitness of an
            # allele at frequency p is 1 + s(1-p) (heterozygote advantage)
            for g in range(g0):
                N = min(2.0 * gamma ** g, ne)
                p = counts / counts.sum()
                pw = p * (1.0 + s_over * (1.0 - p))
                pw = pw / pw.sum()
                counts = rng.multinomial(max(int(2 * N), 2), pw)
            # remaining expansion + plateau: lineage extinction hazard
            # suppressed by balancing selection (Takahata 1990 scaling)
            p_surv = np.exp(-(T0 - g0) * s_over / (2 * ne))
            for k in range(4):
                if counts[k] > 0 and rng.random() > p_surv:
                    counts[k] = 0
            surv_counts.append(int((counts > 0).sum()))
        return float(np.mean(surv_counts))

    ret = {}
    for s_over in [0.01, 0.02, 0.05, 0.10]:
        ret[s_over] = retention_sim(s_over)
        print(f"  overdominance s={s_over:.2f} (gamma=1.6): mean founder lineages "
              f"retained = {ret[s_over]:.2f} of 4")
    for s_over in [0.02, 0.05]:
        v = retention_sim(s_over, gamma=2.0)   # doubling-per-generation schedule
        ret[(s_over, "g2")] = v
        print(f"  overdominance s={s_over:.2f} (gamma=2.0): mean founder lineages "
              f"retained = {v:.2f} of 4")
    print("  (coalescent scaling under overdominance: Takahata 1990)")
    # (2) regeneration of allelic richness: new lineages from mutation +
    # gene conversion at an HLA-scale locus under balancing selection
    L_hla = 3.3e3                     # HLA-class coding target (bp) -- DRB1 exon 2 scale
    kappa = 5.0                       # gene-conversion multiplier (MHC intra-locus conversion)
    s_over = 0.05                     # canonical overdominance for the regeneration rate
    # new selected lineages per generation ~ 2*Ne*mu*L*kappa * P(invade|overdominance)
    # invasion probability of a new allele under overdominance ~ 2s/3 (classic result)
    p_inv = 2.0 * s_over / 3.0
    mu_locus = MU * L_hla * kappa
    # integrate over the trajectory (forward N(t))
    gens = np.arange(T0)
    Nt = np.array([min(2.0 * 1.6 ** g, ne) for g in range(G)] +
                  [ne] * (T0 - G))
    new_line = float(np.sum(2 * Nt * mu_locus * p_inv))
    print(f"  HLA-scale locus (L={L_hla:.0f} bp, kappa={kappa:.0f}, s={s_over}): "
          f"expected new allelic lineages over {gens_to_kya(T0):.0f} kyr: "
          f"{new_line:,.0f}")
    print(f"  observed per-population HLA-DRB1 alleles ~ 12-30 (1000G scale); "
          f"observed worldwide DRB1 allelic lineages ~ 100-400")
    with open(f"{OUT}/hla_regeneration.csv", "w") as f:
        f.write("quantity,value\n")
        for key, v in ret.items():
            if isinstance(key, tuple):
                s_over, tag = key
                f.write(f"mean_founder_lineages_retained_s{s_over:.2f}_{tag},{v:.3f}\n")
            else:
                f.write(f"mean_founder_lineages_retained_s{key:.2f},{v:.3f}\n")
        f.write(f"expected_new_lineages_full,{new_line:.1f}\n")
        f.write(f"gene_conversion_kappa,{kappa}\n")
    return dict(mean_retained=ret, new_line=new_line)


# ----------------------------------------------------------------------------
# Section K. Scoping theorem: rate-free depth-ratio tests
# ----------------------------------------------------------------------------
def section_k():
    print("\n== Section K: scoping tests =========================================")
    """Rate-free (clock-free) tests that delimit the progenitorship claim,
    and the descendant-clade extension (DCE) for archaic admixture.

    Test 1 (archaic sex-chromosome depth ratio): if the archaic clades
    descended from the founding couple INCLUDING their uniparental
    compartments, their mtDNA and Y would coalesce with modern lineages at
    T0 and their divergence depths would EQUAL the modern-modern depths
    (ratio 1:1). Observed: archaic mtDNA diverges from modern mtDNA ~2-3x
    deeper than the modern-modern depth (Green 2008: 206 fixed differences
    vs modern-modern max ~60-100); Neanderthal Y TMRCA ~588 kya vs modern
    Y TMRCA ~275 kya -- ratio ~2.1, CI 1.7-2.9 (Mendez 2016). Ratio test
    FAILS -> the archaic uniparental compartments are not couple-descended;
    under the descendant-clade extension they are externally replaced
    lineages (proposed in the literature for the Y: Mendez 2016; for
    mtDNA: Posth 2017, deeply divergent mtDNA introgressed INTO
    Neanderthals). These compartments leave no trace in modern genomes
    (no archaic mtDNA or Y lineages persist in living humans).

    Test 2 (introgression tracts): 2-6% of modern non-African genomes
    derive from Neanderthal/Denisovan sources (Sankararaman 2012/2014;
    Pruefer 2014). Under the descendant-clade extension the archaic clades
    branch from the couple's descendant population at or after T0, so the
    introgressed tracts ARE couple-descended (returning descendant-branch
    DNA). Branching at T0 requires T0 >= ~550 kya, the youngest edge of
    the published direct-clock split estimates (550-765 kya, Pruefer 2014;
    737 kya, Rogers et al. 2020) -- satisfied at the canonical point
    T0 = 550 kya.

    Scopes:
      S1 (dominant progenitorship): couple = sole genetic progenitors of the
         non-introgressed genome: >= 94-98% of modern nuclear DNA. Unconditional.
      S2 (descendant-clade full nuclear coverage): archaic clades branch from
         the couple's descendants; introgressed tracts couple-descended; 100%
         of modern nuclear DNA covered. Requires T0 in [550, 595] kya AND
         externally replaced archaic uniparental compartments.
      S3 (falsified): full coverage including archaic uniparental
         compartments at any T0 -- killed by Test 1 (clock-free).
    """
    rows = [
        ("modern : archaic mtDNA divergence ratio", "1 : 1 predicted (archaic uniparentals inside)",
         "~1 : 2-3 observed (Green 2008: 206 diffs; Posth 2017: deeper HST)",
         "FAILS -> archaic mtDNA replaced (Posth 2017 mechanism)"),
        ("modern : Neanderthal Y divergence ratio", "1 : 1 predicted (archaic uniparentals inside)",
         "ratio ~2.1 (CI 1.7-2.9); 588 vs 275 kya (Mendez 2016)",
         "FAILS -> archaic Y replaced (Mendez 2016 mechanism)"),
        ("introgressed archaic tracts", "couple-descended if archaic clades branch at/after T0",
         "2-6% of non-African genomes (Sankararaman 2012-2016)",
         "DCE at T0=550 kya covers them (split estimates 550-765 kya)"),
        ("modern mtDNA : modern Y MRCA coincidence", "both = T0 (prediction)",
         "mtDNA 150-200 kya vs Y 120-338 kya, overlapping CIs",
         "CONSISTENT (unexplained coincidence in mainstream demography)"),
        ("archaic-branch heterozygosity", "small descendant branches lose the endowment (drift),\ncarry de-novo diversity only",
         "archaic genomes several-fold less diverse than moderns (Meyer 2012; Pruefer 2014)",
         "PREDICTED (Section N sweep)"),
    ]
    print(f"  {'test':<42}{'prediction':<38}{'observation':<45}verdict")
    for r in rows:
        print(f"  {r[0]:<42}{r[1]:<38}{r[2]:<45}{r[3]}")
    with open(f"{OUT}/scoping_tests.csv", "w") as f:
        f.write("test,prediction,observation,verdict\n")
        for r in rows:
            f.write(",".join('"' + x.replace('\n', ' ') + '"' for x in r) + "\n")
    return rows


# ----------------------------------------------------------------------------
# Section L. Empirical and theoretical existence-proof anchors
# ----------------------------------------------------------------------------
def section_l():
    print("\n== Section L: empirical and theoretical anchors =====================")
    rows = [
        ("Nei, Maruyama & Chakraborty 1975", "Evolution 29:1-10",
         "theory", "bottleneck of a single pair with rapid growth loses little diversity"),
        ("Barton & Charlesworth 1984", "Ann Rev Ecol Syst 15:133-164",
         "theory", "single fertilized female + doubling growth retains >50% heterozygosity"),
        ("Slatkin & Excoffier 2012", "Genetics 191:171-181",
         "theory", "two random individuals carry ~75% of population heterozygosity"),
        ("McMullan et al. 2018", "Nat Ecol Evol 2:1000-1008",
         "empirical", "European ash-dieback invasion founded by two highly divergent individuals"),
        ("Kaeuffer et al. 2007", "Proc R Soc B 274:527-533",
         "empirical", "island mouflon population founded by a single pair retains unexpected heterozygosity"),
        ("Xue et al. 2018", "Biol Invasions 20:3297-3314",
         "empirical", "worldwide invasion of rapa whelk despite dramatic genetic bottleneck"),
        ("Szucs et al. 2017", "Ecol Lett 20:436-444",
         "empirical", "founder effects have long-term fitness consequences; colonising populations persist"),
        ("Ewing et al. 2008", "Conservation Biology 22:395-404",
         "empirical", "Mauritius kestrel: single wild pair to 200+ pairs; extreme inbreeding, persisted"),
        ("Jackson et al. 2022", "Conservation Biology 36:e13918",
         "empirical", "pink pigeon: bottleneck to ~10 -> 400+; demographic recovery with genomic erosion"),
        ("von Seth et al. 2022", "BMC Genomics 23:747",
         "empirical", "Chatham black robin: 5 founders, genomic erosion but persistence"),
        ("Hoelzel et al. 2024", "Nat Ecol Evol 8:686-694",
         "empirical", "northern elephant seal: bottleneck ~20-30 to 200k+; low diversity, viable"),
        ("Rogers et al. 2020; Hu et al. 2023", "Sci Adv 6:eaay5483; Science 381:979-984",
         "literature", "severe archaic-era bottlenecks independently inferred (neandersovan Ne~500; ~1,280 breeding individuals 930-813 ka)"),
    ]
    print(f"  {'anchor':<38}{'reference':<52}{'type':<10}finding")
    for r in rows:
        print(f"  {r[0]:<38}{r[1]:<52}{r[2]:<10}{r[3]}")
    with open(f"{OUT}/empirical_anchors.csv", "w") as f:
        f.write("anchor,reference,type,finding\n")
        for r in rows:
            f.write(",".join('"' + x + '"' for x in r) + "\n")
    return rows


# ----------------------------------------------------------------------------
# Section M. Figures and summary
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Section N. Descendant-clade extension: archaic-branch heterozygosity
# ----------------------------------------------------------------------------
def section_n():
    print("\n== Section N: descendant-clade archaic branches ======================")
    """Under the descendant-clade extension (DCE) the Neanderthal/Denisovan
    clades branch from the couple's descendant population at (or shortly
    after) the founding time T0. A descendant branch that stays small
    (effective size N_b) rapidly loses the endowment by drift (retention
    e^{-Lambda_b} with Lambda_b = T_b/(2*N_b)) and carries de-novo diversity
    only: pi_b ~= 2*mu*E[T2|branch] ~= 4*mu*N_b for a constant-size branch,
    higher with bottleneck-then-growth structure inside the branch. The
    modern branch (plateau 3e4) retains 47% of the endowment.

    Prediction: archaic-branch heterozygosity is several-fold below the
    modern value -- as observed (archaic genomes are markedly less diverse
    than present-day humans; Meyer et al. 2012; Pruefer et al. 2014), and
    independently consistent with the neandersovan bottleneck inferred by
    Rogers et al. 2020 (Ne ~ 500).
    """
    T_b = 18000            # branch duration: couple (550 kya) to archaic sample (~100 kya)
    h_endow = 1.576e-3
    rows = []
    # constant-size branches
    for nb in [100, 200, 500, 1000, 2000, 4000, 8000]:
        lam_b = T_b / (2.0 * nb)
        ret = np.exp(-lam_b)                          # endowment retention in the branch
        # E[T2] within the branch: coalescence in-branch (depth 2N average)
        # or survival to the couple (depth T_b + endowment term below)
        E_T2 = 2.0 * nb * (1.0 - ret) + ret * (T_b + 0.75 * h_endow / (2.0 * MU))
        pi_b = 2.0 * MU * E_T2
        rows.append(dict(model="constant", ne_branch=nb, retention=ret,
                         E_T2=E_T2, pi=pi_b))
        print(f"  constant branch Ne={nb:>5}: endowment retention={ret:.1e}, "
              f"pi_branch={pi_b*1e3:.4f}e-3")
    # bottleneck-then-growth branches (neandersovan-style: bottleneck Ne~500 for
    # the first half, then growth to a larger carrying size)
    for (n_bott, n_late, frac) in [(500, 5000, 0.5), (300, 3000, 0.5), (1000, 10000, 0.4)]:
        t1 = int(T_b * frac)
        lam1 = t1 / (2.0 * n_bott)
        lam2 = (T_b - t1) / (2.0 * n_late)
        ret = np.exp(-(lam1 + lam2))
        E_in = 2.0 * n_bott * (1.0 - np.exp(-lam1)) + np.exp(-lam1) * 2.0 * n_late * (1.0 - ret)
        E_T2 = E_in + ret * (T_b + 0.75 * h_endow / (2.0 * MU))
        pi_b = 2.0 * MU * E_T2
        rows.append(dict(model=f"bottleneck({n_bott}->{n_late})", ne_branch=n_bott,
                         retention=ret, E_T2=E_T2, pi=pi_b))
        print(f"  bottleneck branch Ne {n_bott}->{n_late}: retention={ret:.1e}, "
              f"pi_branch={pi_b*1e3:.4f}e-3")
    # modern-branch reference
    me = master_equation(int(round(kya_to_gens(550.0))), 1.6, 3e4, h_endow)
    print(f"  modern branch (plateau 3e4, canonical): pi={me['pi']*1e3:.3f}e-3 "
          f"(endowment retention {me['eLam']:.2f})")
    with open(f"{OUT}/archaic_branch.csv", "w") as f:
        f.write("model,ne_branch,endowment_retention,E_T2_gens,pi\n")
        for r in rows:
            f.write(f"{r['model']},{r['ne_branch']},{r['retention']:.4e},"
                    f"{r['E_T2']:.1f},{r['pi']:.6e}\n")
    return rows


def make_figures(grid_rows, sfs_res, ld_rows, ne_res, clock_rows, h_canon, joint, branch_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8})
    print("\n== Section M: figures ===============================================")

    # fig1: required endowment curve h_req(T0) at gamma=1.6 for Ne_max grid
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    T0_kya = np.linspace(150, 800, 200)
    for ne, col in [(1e4, "#1f77b4"), (3e4, "#d62728"), (1e5, "#2ca02c")]:
        hreq = []
        for t in T0_kya:
            T0g = int(round(kya_to_gens(t)))
            # invert master equation for h
            me0 = master_equation(T0g, 1.6, ne, 0.0)
            depth_req = (E_T2_REQ - (me0["D_below"] + me0["eLam"] * T0g)) / me0["eLam"]
            hreq.append(max(depth_req * 2 * MU / 0.75, 0.0))
        ax.plot(T0_kya, np.array(hreq) * 1e3, color=col, label=f"$N_e^{{\\max}}$={ne:,.0f}")
    ax.axhline(1.37, color="k", ls=":", lw=1)
    ax.text(155, 1.42, "human avg (4.1M SNVs)", fontsize=7)
    ax.axhline(1.9, color="k", ls="--", lw=1)
    ax.text(155, 1.95, "chimpanzee (5.7M SNVs)", fontsize=7)
    ax.axvspan(joint[0], joint[-1], color="orange", alpha=0.18)
    ax.text((joint[0] + joint[-1]) / 2, 2.35, "joint window\n(TMR4A $\\cap$ $\\pi$)",
            ha="center", fontsize=7)
    ax.set_xlabel("bottleneck time $T_0$ (kya)")
    ax.set_ylabel("required endowment $h_{\\rm req}$ ($\\times10^{-3}$)")
    ax.set_ylim(0, 4.5)
    ax.legend(loc="upper right")
    fig.savefig(f"{OUT}/fig1_required_endowment.pdf")
    fig.savefig(f"{OUT}/fig1_required_endowment.png", dpi=150)
    plt.close(fig)

    # fig2: transmission probabilities
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    n = np.arange(2, 13)
    p4 = (1 - 2.0 * 0.5 ** n) ** 2
    ax.plot(n, p4, "o-", color="#d62728")
    for nn, yy in [(2, 0.25), (3, 0.5625), (4, 0.766)]:
        ax.annotate(f"{yy:.2f}", (nn, yy), textcoords="offset points",
                    xytext=(6, -10), fontsize=7)
    ax.set_xlabel("number of children $n$")
    ax.set_ylabel("P(all four alleles transmitted)")
    ax.set_ylim(0, 1.05)
    fig.savefig(f"{OUT}/fig2_transmission.pdf")
    fig.savefig(f"{OUT}/fig2_transmission.png", dpi=150)
    plt.close(fig)

    # fig3: feasibility heat map (T0 x Ne_max) at gamma=1.6, h=h_canon
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    t0s = sorted(set(r["T0_kya"] for r in grid_rows))
    nes = sorted(set(r["ne_max"] for r in grid_rows))
    M = np.full((len(nes), len(t0s)), np.nan)
    for r in grid_rows:
        if abs(r["h"] - h_canon) < 1e-12 and r["gamma"] == 1.6:
            i = nes.index(r["ne_max"]); j = t0s.index(r["T0_kya"])
            M[i, j] = r["pi"] * 1e3
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis",
                   extent=[min(t0s) - 25, max(t0s) + 25, 0, len(nes) - 1])
    ax.set_yticks(range(len(nes)))
    ax.set_yticklabels([f"{v:,.0f}" for v in nes])
    ax.set_xlabel("$T_0$ (kya)")
    ax.set_ylabel("$N_e^{\\max}$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("$\\pi$ ($\\times10^{-3}$)")
    ax.axvspan(joint[0], joint[-1], facecolor="white", alpha=0.35, lw=2,
               edgecolor="orange")
    fig.savefig(f"{OUT}/fig3_feasibility_map.pdf")
    fig.savefig(f"{OUT}/fig3_feasibility_map.png", dpi=150)
    plt.close(fig)

    # fig4: SFS comparison
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    fs = sfs_res["fs"]
    i = np.arange(1, len(fs) + 1)
    ax.loglog(i, fs / fs[0], "o-", ms=3, label="endowed-pair model (550 kya)")
    ax.loglog(i, 1.0 / i, "k--", lw=1, label="constant-size equilibrium $\\propto 1/i$")
    ax.set_xlabel("derived allele count $i$ (of $n=100$)")
    ax.set_ylabel("relative site frequency")
    ax.legend()
    fig.savefig(f"{OUT}/fig4_sfs.pdf")
    fig.savefig(f"{OUT}/fig4_sfs.png", dpi=150)
    plt.close(fig)

    # fig5: LD decay
    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    d = [r["d_kb"] for r in ld_rows]
    ax.semilogx(d, [r["r2_mean"] for r in ld_rows], "o-", label="endowed-pair model")
    ax.axhspan(0.3, 0.5, color="gray", alpha=0.18)
    ax.text(0.11, 0.46, "observed $r^2$(1 kb) range $\\approx$ 0.3-0.5", fontsize=7)
    ax.set_xlabel("distance (kb)")
    ax.set_ylabel("$E[r^2]$")
    ax.legend(loc="upper right")
    fig.savefig(f"{OUT}/fig5_ld.pdf")
    fig.savefig(f"{OUT}/fig5_ld.png", dpi=150)
    plt.close(fig)

    # fig6: Ne profile
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    t_kya = gens_to_kya(ne_res["centers"][:-1])
    ax.plot(t_kya, ne_res["ne_implied"], color="#1f77b4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("backward time (kya)")
    ax.set_ylabel("PSMC-style implied $N_e$")
    ax.axvline(550, color="orange", ls="--", lw=1)
    ax.text(560, ax.get_ylim()[0] * 1.6, "$T_0$=550 kya\n(N=2: invisible)", fontsize=7,
            color="darkorange")
    fig.savefig(f"{OUT}/fig6_ne_profile.pdf")
    fig.savefig(f"{OUT}/fig6_ne_profile.png", dpi=150)
    plt.close(fig)

    # fig7: clock scan
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    ss = [r["s"] for r in clock_rows]
    mt_lo = [r["mt_ci"][0] for r in clock_rows]
    mt_hi = [r["mt_ci"][1] for r in clock_rows]
    y_lo = [r["y_ci"][0] for r in clock_rows]
    y_hi = [r["y_ci"][1] for r in clock_rows]
    ax.fill_between(ss, mt_lo, mt_hi, alpha=0.25, color="#1f77b4", label="mtDNA CI $\\times s$")
    ax.fill_between(ss, y_lo, y_hi, alpha=0.20, color="#d62728", label="Y span $\\times s$")
    ax.axhspan(joint[0], joint[-1], color="orange", alpha=0.25, label="joint window")
    ax.plot(ss, mt_hi, color="#1f77b4", lw=1)
    ax.plot(ss, mt_lo, color="#1f77b4", lw=1)
    ax.set_xlabel("ancestral-rate correction factor $s$ (Scally-Durbin scaling)")
    ax.set_ylabel("uniparental MRCA date (kya)")
    ax.legend(loc="upper left")
    fig.savefig(f"{OUT}/fig7_clock_scan.pdf")
    fig.savefig(f"{OUT}/fig7_clock_scan.png", dpi=150)
    plt.close(fig)

    # fig8: descendant-clade archaic-branch heterozygosity sweep
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    const = [r for r in branch_rows if r["model"] == "constant"]
    bott = [r for r in branch_rows if r["model"] != "constant"]
    nes = [r["ne_branch"] for r in const]
    pis = [r["pi"] * 1e3 for r in const]
    ax.loglog(nes, pis, "o-", color="#1f77b4", label="constant-size branch")
    for r in bott:
        ax.plot([r["ne_branch"]], [r["pi"] * 1e3], "s", color="#d62728")
    ax.plot([], [], "s", color="#d62728", label="bottleneck-then-growth branch")
    ax.axhline(sfs_res["pi_total"] * 1e3, color="#2ca02c", ls="--", lw=1)
    ax.text(110, sfs_res["pi_total"] * 1e3 * 1.15,
            "modern branch (canonical)", fontsize=7, color="#2ca02c")
    ax.axhspan(0.2, 0.4, color="gray", alpha=0.18)
    ax.text(110, 0.42, "observed archaic scale", fontsize=7)
    ax.set_xlabel("archaic-branch effective size $N_b$")
    ax.set_ylabel("branch $\\pi$ ($\\times10^{-3}$)")
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(f"{OUT}/fig8_archaic_branch.pdf")
    fig.savefig(f"{OUT}/fig8_archaic_branch.png", dpi=150)
    plt.close(fig)

    print("  figures written (fig1..fig8, pdf+png)")


def main():
    global OUT, QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    QUICK = args.quick
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    print("=" * 72)
    print("SOLE PROGENITOR MODEL -- Endowed-Pair feasibility suite")
    print("=" * 72)

    A = section_a()
    section_b()
    grid_rows, win = section_d(A["h_canon"])
    joint = section_e(grid_rows, A["h_canon"])
    joint_window = [joint[0], TMR4A_EST + TMR4A_CI]   # continuous window [400, 595]
    clock_rows = section_f(joint_window)
    sfs_res = section_g(A["h_canon"])
    ld_rows = section_h()
    ne_res = section_i()
    hla = section_j()
    section_k()
    section_l()
    branch_rows = section_n()
    make_figures(grid_rows, sfs_res, ld_rows, ne_res, clock_rows, A["h_canon"], joint_window, branch_rows)

    # canonical parameter block
    T0c = int(round(kya_to_gens(550.0)))
    me = master_equation(T0c, 1.6, 3e4, A["h_canon"])
    summary = {
        "model": "Endowed-Pair (EP): sole-progenitor couple, no pre-bottleneck population",
        "canonical": {
            "T0_kya": 550.0, "T0_gens": T0c, "gamma": 1.6, "ne_max": 3e4,
            "h_endowment": A["h_canon"],
            "founder_snvs_million": A["h_canon"] * L_FULL / 1e6,
            "E_T2_gens": me["E_T2"], "pi": me["pi"],
            "Lambda_c": me["Lambda_c"], "D_below_gens": me["D_below"],
            "depth_endowment_gens": me["depth_endow"],
        },
        "constraints": {
            "pi_observed": PI_OBS,
            "E_T2_required_gens": E_T2_REQ,
            "TMR4A_kya": [TMR4A_EST, TMR4A_CI],
            "joint_window_kya": joint_window,
            "descendant_clade_window_kya": [550.0, joint_window[1]],
        },
        "sfs": {"pi_total": sfs_res["pi_total"], "tajimas_D": sfs_res["D"],
                "pi_backbone": sfs_res["pi_bb"]},
        "hla": {str(k) if not isinstance(k, tuple) else f"{k[0]}_{k[1]}": v
                for k, v in hla["mean_retained"].items()
                } | {"new_line": hla["new_line"]},
        "runtime_sec": time.time() - t_start,
        "seed_note": "fixed seeds: msprime 20260101+, backbone 99001, LD 88001, HLA 66001, NE 770001+",
    }
    with open(f"{OUT}/feasibility_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # human-readable report
    with open(f"{OUT}/feasibility_report.txt", "w") as f:
        f.write("ENDOWED-PAIR MODEL -- FEASIBILITY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Canonical scenario: T0 = 550 kya ({T0c} generations), "
                f"gamma = 1.6, Ne_max = 30,000,\n")
        f.write(f"endowment h = {A['h_canon']:.2e} per site "
                f"({A['h_canon']*L_FULL/1e6:.2f}M SNVs per founder; within the\n"
                f"primate range bounded by human 3.1-4.1M and chimpanzee 5.7M SNVs).\n\n")
        f.write(f"Master equation:  E[T2] = D_below + e^(-Lam_c) * [T0 + 0.75*h/(2*mu)]\n")
        f.write(f"  Lambda_c = {me['Lambda_c']:.3f};  D_below = {me['D_below']:,.0f} gen; "
                f"e^-Lam_c = {me['eLam']:.3f}\n")
        f.write(f"  E[T2] = {me['E_T2']:,.0f} generations = "
                f"{gens_to_kya(me['E_T2']):.2f} kya\n")
        f.write(f"  pi = 2*mu*E[T2] = {me['pi']*1e3:.3f}e-3  "
                f"(observed 1.0e-3; required E[T2] {E_T2_REQ:,.0f} gen)\n\n")
        f.write(f"TMR4A compression identity: TMR4A_true = T0 = 550 kya "
                f"(Swamidass estimate 495 +/- 100 kya, direct clock).\n")
        f.write(f"Joint feasibility window (pi feasible AND TMR4A CI): "
                f"{joint_window} kya, nominal 550 kya.\n")
        f.write(f"Descendant-clade window (archaic clades branch at/after founding): "
                f"T0 in [550, {joint_window[1]:.0f}] kya -- available at the canonical point.\n\n")
        f.write(f"SFS: pi_total = {sfs_res['pi_total']*1e3:.3f}e-3, "
                f"theta_W = {sfs_res['thetaW_total']*1e3:.3f}e-3, "
                f"Tajima's D = {sfs_res['D']:.2f}\n")
        f.write("  (observed: pi ~ 1.0e-3; strongly negative D ~ -2 in large-sample\n")
        f.write("   series, Tennessen et al. 2012)\n\n")
        f.write("Scoping: S1 dominant progenitorship (>=94-98% of nuclear DNA) "
                "unconditional;\n")
        f.write("S2 descendant-clade full nuclear coverage at T0 in [550,595] kya "
                "(archaic clades\n")
        f.write("branch from the couple's descendants; archaic uniparentals "
                "externally replaced\n")
        f.write("per Mendez 2016 / Posth 2017 mechanisms); S3 (archaic "
                "uniparentals included)\n")
        f.write("falsified by the clock-free depth-ratio test (Section K).\n\n")
        f.write(f"Runtime: {time.time()-t_start:.1f} s (quick={QUICK})\n")

    print("\n== outputs written to", OUT, "==")
    for fn in sorted(os.listdir(OUT)):
        print("   ", fn)
    print(f"\nTOTAL runtime {time.time()-t_start:.1f} s")


if __name__ == "__main__":
    main()
