# scc — patient-side simulator and consultation environment

State-channel confounding project. The patient is an *uncalibrated channel*: every fact has a true value and a reported value, and
the same channel rules that distort reports are used by the oracle to compute the exact posterior P(Z, U | history).

## Install
```
pip install -r requirements.txt          # openai httpx pydantic pyyaml numpy pandas matplotlib pytest scikit-learn
cp pilot/.env.example pilot/.env         # relay URL + key (only needed for --components llm)
pytest tests -q                          # 119 tests, no API calls
```

## Run (no LLM)
```
python run_episode.py --config configs/m1_stub.yaml               # 11 cases x 4 channels x 2 seeds, scripted doctor
python -m scc.sim.audit results/episodes/m1_stub                  # recompute every report value from the rules
python -m scc.analysis.trajectory results/episodes/m1_stub        # trajectories.png
python run_episode.py --config configs/m1_stub.yaml --cases cp_001 --channels "exaggerate:k=2,flooding_rate=0.3+self_dx;vague:p=0.8" --dry-run
```
## Run (LLM phrasing; needs relay)
```
python run_episode.py --config configs/m2_smoke.yaml              # ~20 calls
```

## Doctors (the other side of the environment)
```
python run_episode.py --config configs/d_m1_stub.yaml        # tested doctor with a fake LLM (replayed script), 0 API
python run_episode.py --config configs/d_m2_smoke.yaml       # tested doctor = qwen3.7-plus, Vanilla baseline
python run_episode.py --config configs/d_m3_agent.yaml       # scaffold agent, stub components (regex perceiver, menu candidates, table likelihood)
python run_episode.py --config configs/d_m4_agent_smoke.yaml # scaffold agent, LLM perceiver / candidates / LLM-estimated medical table
python -m scc.analysis.compare results/episodes/d_m3_agent results/episodes/d_m3_agent_zonly   # comparison table
python -m scc.doctor.rl.train_min --config configs/d_m5_rl_dry.yaml   # minimal GRPO dry run (0.5B policy, stub patient, 0 API)
```
Doctor types in a config: `{type: api, baseline: vanilla|cot|uncertainty|told}`, `{type: agent, components: stub|llm, likelihood: table|llm, z_only, lookahead, anchoring, stopping, monitor}`, scripted doctors, or `type: "module:Class"`.
Ablations of the agent are configuration switches; `z_only: true` is the key one (no channel hypotheses).

## Layout
```
scc/types.py        data structures + doctor<->env contract
scc/env/            cases.py (load/validate), channels.py (6 kernels, report table, likelihood_kernel), likelihood.py, oracle.py
scc/env/clusters/chest_pain/   cluster.yaml slots.yaml vocab.yaml likelihood.yaml cases/*.json
scc/sim/            patient.py rules.py classifier.py phraser.py checker.py persona.py router.py episode.py scripted_doctor.py fidelity.py audit.py
scc/doctor/base.py  Doctor protocol, parse_doctor_text, format_action_protocol
scc/analysis/       trajectory.py metrics.py
scc/data/           explore/estimate/build (DDXPlus), mine_style (IMCS-21 / CPB-Bench)
```

## Plugging in a doctor (the only thing the doctor side must do)
1. Implement `Doctor` (`scc/types.py`): `name`, `reset(ctx)`, `act(ctx) -> DoctorAction`; add a `readout` when you can.
2. Build the action-format text for your prompt with `scc.doctor.base.format_action_protocol(lang, ctx.ddx_hint)`;
   convert model text with `scc.doctor.base.parse_doctor_text(text)`.
3. Read only `DoctorContext` (case brief + transcript). Tests assert the doctor cannot see the `Case`.
4. Register it in `run_episode.py` via `doctor: {type: "scc.doctor.my_doctor:MyDoctor", kwargs: {...}}`.
5. Metrics come from `scc/analysis/metrics.py` over the JSONL logs; do not write your own logs.

Log format (one JSONL row per turn): see `PATIENT_CODE_PLAN_v3.md` §2.6.
