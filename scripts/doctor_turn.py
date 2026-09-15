"""One doctor turn at a time against the LLM patient; env state is pickled between calls.

  python scripts/doctor_turn.py --session A --start cp_001 "exaggerate:k=2,flooding_rate=0.3"   # opening
  python scripts/doctor_turn.py --session A "Where exactly is the pain?"                           # one turn
  python scripts/doctor_turn.py --session A "VERIFY RECORD: prior_severity"
  python scripts/doctor_turn.py --session A "DIAGNOSIS READY: GERD"
"""
import argparse, json, pickle, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scc.types import *
from scc.env.cases import load_cluster, load_cluster_config
from scc.env.likelihood import Likelihood
from scc.env.oracle import Oracle
from scc.sim.components import llm_components
from scc.sim.episode import ConsultationEnv
from scc.doctor.base import parse_doctor_text
from scc.llm import ChatClient
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); from run_episode import parse_channel_arg  # noqa

OUT = pathlib.Path("results/interactive")


def fmt_oracle(o):
    z = o["posterior_z"]; u = o["posterior_u"]
    top = sorted(z.items(), key=lambda kv: -kv[1])[:3]
    return "oracle z: " + ", ".join(f"{k} {v:.2f}" for k, v in top) + " | u: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(u.items(), key=lambda kv: -kv[1])[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True); ap.add_argument("--start", nargs=2, metavar=("CASE", "CHANNELS"), default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("text", nargs="?", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pkl = OUT / f"{a.session}.pkl"; md = OUT / f"{a.session}.md"
    cfg = load_cluster_config("chest_pain")
    if a.start:
        case = load_cluster("chest_pain", [a.start[0]])[0]
        ch = parse_channel_arg(a.start[1])
        pc = PatientConfig(channels=[ChannelSpec.from_dict(s) for s in ch["specs"]], seed=a.seed, name=ch["name"])
        comp = llm_components(cfg, ChatClient("qwen3.7-plus", temperature=0.7, use_cache=False), ChatClient("qwen3.7-max", temperature=0.0), "regex", "template")
        env = ConsultationEnv(case, pc, comp, cfg, Oracle(case, cfg, Likelihood(cfg)), max_turns=20, lang="en",
                              log_path=OUT / f"{a.session}.jsonl", episode_id=f"{case.case_id}|{pc.name}|{pc.persona.code()}|seed={a.seed}|doctor=interactive", doctor_name="interactive")
        obs, ctx = env.reset()
        md.write_text(f"# Session {a.session}: {case.case_id} ({case.labels.primary_dx}) | channel {pc.name} | seed {a.seed}\n\nDoctor: manual, one question typed per turn. Patient: qwen3.7-plus phrasing, qwen3.7-max classifier, regex checker.\n\n"
                      f"**Brief given to doctor:** {ctx.case_brief}\n\n**Patient (opening):** {obs.text}\n\n")
        print(f"[brief] {ctx.case_brief}\n[patient] {obs.text}\n[{fmt_oracle(env.turn_logs[-1].oracle)}]")
    else:
        env = pickle.load(open(pkl, "rb"))
        action = parse_doctor_text(a.text)
        obs, ctx, log, done = env.step(action)
        disc = ", ".join(f"{d['atom']}={d['report_value']}" + ("" if d["report_value"] == d["true_value"] else f" (true {d['true_value']})") for d in log.disclosed)
        line = (f"**Doctor (t{log.turn}):** {a.text}\n\n**{obs.kind.value.title()}:** {obs.text}\n\n"
                f"<sub>form={log.question_form} hit={log.hit_atoms} disclosed=[{disc}] withheld={log.withheld} checker={'ok' if log.checker['pass'] else log.checker['violations']} retries={log.checker['retries']} | {fmt_oracle(log.oracle)}</sub>\n\n")
        with open(md, "a") as f:
            f.write(line)
        print(f"[patient/{obs.kind.value}] {obs.text}\n[form={log.question_form} hit={log.hit_atoms} disclosed=[{disc}] withheld={log.withheld} checker={'ok' if log.checker['pass'] else log.checker['violations']}]\n[{fmt_oracle(log.oracle)}]")
        if done:
            calls = {k: v.stats() for k, v in {"conv": env.patient.comp.phraser.client, "meta": env.patient.comp.classifier.client}.items()}
            with open(md, "a") as f:
                f.write(f"\n**Episode ended.** LLM calls: conversational {calls['conv']['calls']}, meta {calls['meta']['calls']} (cache hits {calls['meta']['cache_hits']}).\n")
            print("[done]", calls)
    pickle.dump(env, open(pkl, "wb"))


if __name__ == "__main__":
    main()
