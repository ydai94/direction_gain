"""Emit measured run report and append a distinct canonical experiment entry."""
from common import CONFIG, ROOT, OUT, cases, read_json, digest

def main():
    r=read_json(OUT/"results.json");v=read_json(OUT/"verification.json")
    if not v["passed"] or v["results_sha256"] != digest(OUT/"results.json"):
        raise AssertionError("verified results required")
    rows=[]
    for arm in (r["selected"]["F"],r["selected"]["M"],r["selected"]["A"],"F2","A2","C"):
        rows.append("| "+arm+" | "+" | ".join(f"{r['summary'][arm][m]['mean']:.5f}" for m in ("Y","H","J","P","Q","Q_visible"))+" |")
    contrast=[]
    for family,c in r["evaluation_contrasts"].items():
        for metric in ("J","P","Q","Y_minus_clean"):
            e=c[metric]; contrast.append(f"- {family} minus tuned full, {metric}: {e['mean']:+.5f} [{e['ci95'][0]:+.5f}, {e['ci95'][1]:+.5f}]." if metric != "Y_minus_clean" else f"- {family} minus clean, Y: {e['mean']:+.5f} [{e['ci95'][0]:+.5f}, {e['ci95'][1]:+.5f}].")
    text=f"""# Exp265A output dose and preservation report

Mode: measured. Date: 2026-09-06. Data contract: data_contract.md.
Decision: **{r['decision']}**. This is an exploratory development screen, not prospective confirmation.

## Problem

The experiment asks whether weak anchor efficacy is an operating-dose problem, a write-support limitation, or a difference in direction construction. Earlier comparisons mixed these factors. Full positional output used the complete position-dependent anti-minus-stereotype difference; the local operator wrote a normalized pooled vector at subject rows. Raising the local dose might improve efficacy, but improvement alone would not establish an advantage if full output can obtain the same preservation simply by reducing its own strength. This run therefore includes three operators and a matched writing-budget ladder, with the output site held fixed throughout.

## Design

Twenty-four previously reviewed reliable anchors were selected by the frozen text-only source-by-bias_type procedure. A separate deterministic hash split allocated twelve cases to calibration and twelve to evaluation. All prompts are historically exposed development material. Image seeds 3, 4 and 5 were paired across all methods. New seeds improve the scope of observed stochastic replication but do not remove earlier prompt exposure or turn the evaluation subset into a confirmatory holdout. Every case has fifteen matched-budget conditions, two exact legacy references and one unedited control, totaling 1,296 new images. There is no outcome-conditioned sample replacement or best-seed reporting.

The permitted target changes were specified in text before model execution. This is necessary because a change of age legitimately changes some visible identity cues, while changing occupation or activity can require different tools or gestures. Only those specific target-related changes were exempted. Arbitrary replacement of subjects, counts or backgrounds was not excused. Several source contrasts concern abstract intent or disposition, so an independent visible-target judgment can return unjudgeable rather than infer those attributes from appearance.

## Method

The full positional operator F preserves the established padding and truncation convention. The masked positional operator M keeps the same direction entries only at the frozen anchor rows. Pooled anchor A writes the same pooled pole direction at those rows. All three shapes receive tensor Frobenius budgets proportional to the raw pooled-anchor norm, over the fixed 128-fold range. A bounded scalar search matches actual BF16 displacement norms to within five percent of each requested value, using no behavioral outcome or finite-N score. Actual norm, requested norm, edited support, effective multiplier and conditioning hashes are retained. This norm control does not claim equal semantic effect or tokenwise semantic alignment.

Generation uses Qwen-Image with 50 steps, CFG four and a one-space negative prompt. Model parameters are frozen. Finite N is measured independently on six neutral-trajectory cells using seeds 6, 7 and 8 at indices 10 and 16. It projects the conditional velocity change onto the anti-minus-stereotype reference response. It is neither a full edited-rollout measurement nor a direct image-quality objective. No N-based dose selection, layer selection or token-weight optimization occurs in this experiment.

## Results

Calibration-selected arms: {r['selected']}. Rates below refer to the twelve evaluation cases, averaging three image seeds within each case.

| Arm | Y | H | J | P | Q | Q_visible |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

{chr(10).join(contrast)}

The intervals are pointwise descriptive paired case-bootstrap intervals from 10,000 fixed-seed resamples. They are not a familywise guarantee over every dose or endpoint. The complete grid, per-seed rates, frozen selected arms and numerical provenance are in results.json and case_metrics.parquet. Figure 1 (figures/fig1_dose_curves.pdf) separates calibration and evaluation efficacy and preservation curves. Figure 2 (figures/fig2_efficacy_preservation.pdf) places each fixed dose in efficacy-preservation space. Figure 3 (figures/fig3_norm_response.pdf) relates achieved writing norm to the finite response proxy without claiming semantic causation. Publication acceptance still requires visual inspection of the rendered figure exports.

## Analysis

The decision rule checks whether either selected local operator retains J within 0.05 of calibration-tuned full output, improves absolute preservation by at least 0.10 and joint preserved success Q by at least 0.05, and improves target choice by at least 0.10 over clean. Each condition is an exploratory practical threshold. Meeting all conditions would justify consideration of a separately registered follow-up; it would not launch that follow-up or prove a general advantage. Failing the screen closes this tested configuration and finite dose grid without establishing impossibility at other directions, strengths or subjects.

Quality diagnostics: clean-control P={r['quality']['clean_P']:.5f}; preservation unjudgeability={r['quality']['unjudgeable_fraction']:.5f}; target unjudgeability={r['quality']['target_unjudgeable_fraction']:.5f}; repeated-order P agreement={r['quality']['repeat_order_P_agreement']:.5f} across {r['quality']['n_repeats']} repeated pairs. Structural-zero masked positional cases: {len(r['quality']['masked_structural_zero_cases'])}. These execute zero edit and do not attain a nonzero matched budget; they remain in endpoint denominators. If the registered measurement gates fail, preservation cannot support an advance even when point estimates appear favorable. The clean baseline and Q_visible diagnostic also reveal cases where coarse forced choice scores a target that was already present or not visibly established.

Comparison of F and M controls direction construction and requested writing budget while changing support. Comparison of M and A controls anchor support and budget while changing direction geometry. Neither comparison establishes that individual token entries encode pure attributes, because positions are aligned mechanically rather than semantically. The finite-N results can describe response sensitivity, but an N increase with no improvement in J or Q does not rescue the intervention. All outcomes remain distinct from a claim that backward chooses useful weights: no such optimizer was tested here.

## Limitations

The twelve evaluation cases provide limited precision and represent only prompts with already accepted explicit anchors. The cohort selected here contains StereoSet material only, so cross-source generality is not established. Both calibration and evaluation prompts were exposed in earlier development. Only three image seeds and one generator were evaluated. Automated Qwen3-VL judgments are not human validation and may share model-family preferences. Abstract targets and target-exemption wording can affect measured success. Equal native tensor norms control one engineering dimension but do not equate perceptual intervention size. No deployable prompt-only method was evaluated: the sample's own counterfactual pair defines its direction, dose reference and probe target. Large or small intervals should retain this scope when incorporated into the paper.

## Reproduction

Use the frozen code and config under pipelines/layer_routing/finite_n, the bound cases.json and original source hash, and the cluster project environment. Run the nonbenchmark smoke, per-case generation, image verification, per-case finite probes, per-case scoring, and CPU finalization with the saved Slurm dependency graph. Every score retains raw judge attempts and each image is hash-verified. analysis uses a fixed bootstrap seed; verify.py independently reconstructs endpoint means and paired intervals. The saved per-case payloads contain job IDs and execution times, and the immutable run archive must include all images, logs, probes and code snapshots. Do not use this report as proof of scheduler completion without the accompanying sacct checks and final verification artifact.
"""
    (OUT/"experiment_report.md").write_text(text)
    heading="## Exp265A — output dose and preservation"
    canonical=ROOT/"results/REPORT.md"
    if heading not in canonical.read_text():
        with canonical.open("a") as stream:
            stream.write("\n\n"+heading+"\n\n"+text.removeprefix("# Exp265A output dose and preservation report\n"))

if __name__ == "__main__":
    main()
