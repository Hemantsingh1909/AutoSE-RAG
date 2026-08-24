#!/usr/bin/env python3
"""
CLI entry point to execute the PhD comparative benchmark suite.
Outputs:
- Experiment 1: Information Retrieval with 95% Bootstrap CIs
- Experiment 2: Software Synthesis & Independent Oracle Verification with Paired Significance Tests (Cohen's d_z & Holm Correction)
- Per-Requirement Diagnostic Trace Table (Failure Taxonomy F1-F7 & Repair Actions)
- Component-Wise Ablation Matrix (A0 to A6)
- Publication-Ready LaTeX Tables
- Structured JSON Artifacts in results/
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.eval.experiment_runner import ExperimentRunner

def main():
    print("=" * 84)
    print("AutoSE-RAG: PhD Comparative Research Benchmark Suite (15 Requirements)")
    print("=" * 84)

    runner = ExperimentRunner()

    print("\n[+] Running Experiment 1: Information Retrieval Evaluation (Sparse vs Dense vs Hybrid)...")
    exp1 = runner.run_retrieval_experiment(max_items=15)
    print("\n### Experiment 1 Results (Information Retrieval with 95% Bootstrap CI):\n")
    print(exp1["markdown_table"])

    print("\n" + "=" * 84)
    print("[+] Running Experiment 2: Software Synthesis & Independent Oracle Verification...")
    exp2 = runner.run_generation_experiment(max_items=15)
    print("\n### Experiment 2 Results (Software Synthesis & Oracle Verification with 95% CI):\n")
    print(exp2["markdown_table"])

    print("\n### Paired Statistical Significance & Effect Sizes (Traceability Completeness, Holm Corrected):")
    sig = exp2["significance"]
    s21 = sig["traceability_m2_vs_m1"]
    s20 = sig["traceability_m2_vs_m0"]
    s10 = sig["traceability_m1_vs_m0"]
    print(f"  - M2 (Agentic) vs M1 (Standard RAG): t = {s21['t_stat']}, Cohen's d_z = {s21['cohens_dz']}, Holm-adj p = {s21['holm_adj_p']:.6f} (Significant: {s21['significant']})")
    print(f"  - M2 (Agentic) vs M0 (Raw LLM):      t = {s20['t_stat']}, Cohen's d_z = {s20['cohens_dz']}, Holm-adj p = {s20['holm_adj_p']:.6f} (Significant: {s20['significant']})")
    print(f"  - M1 (Standard RAG) vs M0 (Raw LLM): t = {s10['t_stat']}, Cohen's d_z = {s10['cohens_dz']}, Holm-adj p = {s10['holm_adj_p']:.6f} (Significant: {s10['significant']})")

    print("\n" + "=" * 84)
    print("### Per-Requirement Diagnostic Trace & Failure Taxonomy Table (F1 to F7):\n")
    print(exp2["trace_table"])

    print("\n" + "=" * 84)
    print("### Component-Wise Ablation Study Matrix (A0 to A6):\n")
    print(exp2["ablation_table"])

    print("\n" + "=" * 84)
    print("### Publication-Ready LaTeX Tables:\n")
    print(exp1["latex_table"])
    print("\n")
    print(exp2["latex_table"])

    print(f"\n[+] Raw reproducible JSON outputs saved to:")
    print(f"    - results/retrieval_results.json")
    print(f"    - results/generation_results.json")
    print(f"    - results/statistical_results.json")
    print(f"    - results/raw/R01.json ... R15.json")

    print(f"\n[+] Visualization charts generated in static/:")
    print(f"    - {exp1['chart_path']}")
    print(f"    - {exp2['chart_path']}")

if __name__ == "__main__":
    main()
