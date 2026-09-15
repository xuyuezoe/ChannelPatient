"""J2 (post-integration): LLM phraser rule adherence on 4 channels x 5 plans x 3 samples (~60 API calls).
  python tests/eval_phraser.py
"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scc.env.cases import load_cluster, load_cluster_config
from scc.env.channels import build_report_table
from scc.sim.rules import RuleEngine
from scc.sim.phraser import LLMPhraser
from scc.sim.checker import RegexChecker
from scc.sim.persona import TemplateRole
from scc.types import *

PLANS = [(["cp_location"], "open"), (["cp_severity", "cp_functional"], "severity_open"), (["assoc_sweating", "assoc_dyspnea"], "yes_no"), (["cp_quality"], "forced_choice"), (["assoc_acid"], "open")]
CHANNELS = {"cooperative": [], "exaggerate_k2": [ChannelSpec("exaggerate", {"k": 2, "flooding_rate": 1.0})], "vague_p08": [ChannelSpec("vague", {"p": 1.0})], "self_dx": [ChannelSpec("self_dx", {"label": "stable_angina", "conviction": "insist"})]}

if __name__ == "__main__":
    from scc.llm import ChatClient
    cfg = load_cluster_config("chest_pain"); case = load_cluster("chest_pain", ["cp_001"])[0]
    ph = LLMPhraser(cfg, ChatClient("qwen3.7-plus", temperature=0.7, use_cache=False)); ck = RegexChecker(cfg); role = TemplateRole().build(Persona(), case, "en")
    rows = []; ok = 0; n = 0
    for name, specs in CHANNELS.items():
        pc = PatientConfig(channels=specs, seed=0); t, sd, _ = build_report_table(case, pc, cfg); r = RuleEngine(case, t, sd, pc, cfg)
        for hits, form in PLANS:
            plan = r.plan(hits, form, 1)
            for k in range(3):
                utt = ph.phrase(plan, role, [("doctor", "?")], "en"); res = ck.check(utt, case=case, plan=plan, lang="en")
                n += 1; ok += res.passed
                rows.append({"channel": name, "plan": hits, "form": form, "utterance": utt, "pass": res.passed, "violations": res.violations, "say": [(a.id, v) for a, v in plan.say], "withheld": [a.id for a in plan.withheld]})
    out = pathlib.Path("results/eval_phraser.md")
    out.write_text("# 措辞器评测\n\n" + f"规则遵从：{ok}/{n} = {ok/n:.3f}\n\n" + "\n".join(f"- [{r['channel']}] {r['form']} say={r['say']} withheld={r['withheld']}\n  → {r['utterance']}  {'OK' if r['pass'] else r['violations']}" for r in rows) + "\n")
    print(f"rule adherence {ok}/{n}", "->", out)
