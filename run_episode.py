"""CLI: run one or many consultations (patient side) against a doctor.

  python run_episode.py --config configs/m1_stub.yaml [--cases cp_001] [--channels "exaggerate:k=2,flooding_rate=0.3;vague:p=0.8"] [--seeds 0]
                        [--doctor scripted_fixed] [--components stub|llm] [--lang en] [--max-turns 12] [--dry-run] [--workers 4]
Outputs results/episodes/<run_id>/<episode>.jsonl + manifest.json (+ failed.jsonl).
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
from scc.config import settings, ROOT
from scc.types import *
from scc.env.cases import load_cluster, load_cluster_config, validate
from scc.env.likelihood import Likelihood
from scc.env.oracle import Oracle
from scc.sim.episode import run_episode
from scc.sim.components import stub_components, llm_components
from scc.sim import scripted_doctor as SD


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "n/a"


def make_doctor(spec: dict | str, cfg, lang: str):
    if isinstance(spec, str):
        spec = {"type": spec}
    t = spec.get("type", "scripted_fixed")
    if t == "scripted_fixed":
        qs = spec.get("questions")
        if isinstance(qs, str) and qs.startswith("default"):
            qs = None
        return SD.FixedListDoctor(qs, lang)
    if t == "scripted_reask":
        return SD.ReaskDoctor(cfg, lang)
    if t == "scripted_induce":
        return SD.InduceDoctor(lang)
    if t == "scripted_transcript":
        d = json.load(open(spec["file"]))[spec["dialog_id"]]
        return SD.TranscriptDoctor(d)
    # doctor-side implementations register here later (scc.doctor.<module>.<Class>)
    if ":" in t:
        mod, cls = t.split(":")
        import importlib
        return getattr(importlib.import_module(mod), cls)(**spec.get("kwargs", {}))
    raise ValueError(f"unknown doctor type {t}")


def parse_channel_arg(s: str) -> dict:
    """'exaggerate:k=2,flooding_rate=0.3+self_dx' -> config entry (channels on the CLI are separated by ';')"""
    specs = []
    for part in s.split("+"):
        kind, _, ps = part.partition(":")
        params = {}
        for kv in filter(None, ps.split(",")):
            k, v = kv.split("=")
            try:
                v = json.loads(v)
            except Exception:
                pass
            params[k] = v
        if kind != "cooperative":
            specs.append({"kind": kind, "params": params})
    return {"name": s, "specs": specs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--cluster", default=None); ap.add_argument("--cases", default=None); ap.add_argument("--channels", default=None)
    ap.add_argument("--seeds", default=None); ap.add_argument("--doctor", default=None); ap.add_argument("--components", default=None)
    ap.add_argument("--lang", default=None); ap.add_argument("--max-turns", type=int, default=None); ap.add_argument("--run-id", default=None)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--patient-model", default=None); ap.add_argument("--meta-model", default=None); ap.add_argument("--checker", default=None)
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()
    cfgd = yaml.safe_load(open(args.config)) if args.config else {}
    cluster = args.cluster or cfgd.get("cluster", "chest_pain")
    lang = args.lang or cfgd.get("lang", "en")
    max_turns = args.max_turns or cfgd.get("max_turns", 12)
    run_id = args.run_id or cfgd.get("run_id", f"run_{int(time.time())}")
    comp_kind = args.components or cfgd.get("components", "stub")
    models = dict(cfgd.get("models", {}))
    if args.patient_model: models["conversational"] = args.patient_model
    if args.meta_model: models["meta"] = args.meta_model
    if args.checker: models["checker"] = args.checker
    cfg = load_cluster_config(cluster)
    L = Likelihood(cfg)
    cases = load_cluster(cluster)
    sel = args.cases or cfgd.get("cases", "all")
    if sel != "all":
        ids = sel.split(",") if isinstance(sel, str) else list(sel)
        cases = [c for c in cases if c.case_id in ids]
    for c in cases:
        errs = validate(c, cfg)
        if errs:
            raise SystemExit(f"case {c.case_id} invalid: {errs}")
    channels = [parse_channel_arg(s) for s in args.channels.split(";")] if args.channels else cfgd.get("channels", [{"name": "cooperative", "specs": []}])
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfgd.get("seeds", [0]))
    persona = Persona.from_dict(cfgd.get("persona", {}))
    doctor_spec = args.doctor or cfgd.get("doctor", "scripted_fixed")
    oracle_cfg = cfgd.get("oracle", {})
    out_dir = settings.results_dir / "episodes" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    conv_client = meta_client = None
    if comp_kind == "llm":
        from scc.llm import ChatClient
        conv_client = ChatClient(models.get("conversational", settings.default_models["conversational"]), temperature=0.7)
        meta_client = ChatClient(models.get("meta", settings.default_models["meta"]), temperature=0.0)

    jobs = []
    for case in cases:
        for ch in channels:
            for seed in seeds:
                pc = PatientConfig(persona=persona, channels=[ChannelSpec.from_dict(s) for s in ch.get("specs", [])], seed=seed,
                                   post_verify_shift=int(cfgd.get("post_verify_shift", 0)), sample_params_from_persona=bool(cfgd.get("sample_params_from_persona", False)), name=ch["name"])
                jobs.append((case, pc))
    if args.max_episodes:
        jobs = jobs[: args.max_episodes]
    print(f"[run {run_id}] {len(jobs)} episodes, components={comp_kind}, doctor={doctor_spec}, lang={lang}", flush=True)

    if args.dry_run:
        from scc.env.channels import build_report_table
        for case, pc in jobs[:8]:
            t, sd, _ = build_report_table(case, pc, cfg)
            print(f"\n== {case.case_id} | {pc.label()} | seed={pc.seed} | self_dx={sd}")
            for aid, e in t.items():
                if e.rule != "identity":
                    print(f"   {aid:16s} {str(e.true_value)[:30]:30s} -> {str(e.report_value)[:30]:30s} [{e.rule}] disc={e.disclosure}")
        return

    manifest = {"run_id": run_id, "config": cfgd, "cli": vars(args), "git": git_commit(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_episodes": len(jobs), "components": comp_kind, "models": models if comp_kind == "llm" else {}, "failed": []}
    failed_path = out_dir / "failed.jsonl"

    def one(job):
        case, pc = job
        comp = stub_components(cfg, pc.seed) if comp_kind == "stub" else llm_components(cfg, conv_client, meta_client, models.get("checker", "regex"), cfgd.get("role", "template"))
        doctor = make_doctor(doctor_spec, cfg, lang)
        eid = f"{case.case_id}|{pc.name}|{pc.persona.code()}|seed={pc.seed}|doctor={doctor.name}"
        fname = out_dir / (eid.replace("|", "__").replace("=", "-") + ".jsonl")
        oracle = Oracle(case, cfg, L, u_library={k: [ChannelSpec.from_dict(s) for s in v] for k, v in oracle_cfg["u_library"].items()} if isinstance(oracle_cfg.get("u_library"), dict) else None,
                        prior_u=oracle_cfg.get("prior_u"))
        try:
            logs = run_episode(case, pc, doctor, comp, cfg, oracle, max_turns, lang, fname, eid)
            return eid, len(logs), None
        except Exception as e:
            with open(failed_path, "a") as f:
                f.write(json.dumps({"episode": eid, "error": repr(e), "trace": traceback.format_exc()[-2000:]}) + "\n")
            return eid, 0, repr(e)

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for eid, n, err in ex.map(one, jobs):
            done += 1
            status = f"FAILED {err}" if err else f"{n} turns"
            print(f"  [{done}/{len(jobs)}] {eid}: {status}", flush=True)
            if err:
                manifest["failed"].append(eid)
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S"); manifest["elapsed_s"] = round(time.time() - t0, 1)
    if conv_client:
        manifest["llm"] = {"conversational": conv_client.stats(), "meta": meta_client.stats()}
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2, ensure_ascii=False)
    print(f"[run {run_id}] done in {manifest['elapsed_s']}s, failed={len(manifest['failed'])}, out={out_dir}")


if __name__ == "__main__":
    main()
