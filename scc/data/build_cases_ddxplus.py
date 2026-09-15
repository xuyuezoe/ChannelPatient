"""Build chest-pain cases from the DDXPlus test split (no API calls).

For each dx in cluster.yaml with a DDXPlus condition: sample patients, map evidences -> atoms via slots.yaml,
fill slots DDXPlus lacks from per-dx templates (below), write cases/cp_NNN.json (source.dataset = ddxplus).
Costochondritis (no DDXPlus condition) is skipped here (handwritten).
  python -m scc.data.build_cases_ddxplus --per-dx 2 --start 4 [--seed 0]
"""
from __future__ import annotations
import argparse, ast, csv, json, random
from pathlib import Path
import yaml
from scc.config import settings, ROOT
from scc.env.cases import load_cluster_config, validate
from scc.types import Case

DDX = ROOT / "data" / "external" / "ddxplus"
LOC_ORDER = ["epigastric", "retrosternal", "precordial", "lateral_chest"]
LOC_PREF = {"GERD": ["epigastric", "retrosternal", "precordial", "lateral_chest"], "stable_angina": ["retrosternal", "precordial", "epigastric", "lateral_chest"],
            "pericarditis": ["retrosternal", "precordial", "lateral_chest", "epigastric"], "panic_attack": ["precordial", "lateral_chest", "retrosternal", "epigastric"]}
AGE_RANGE = {"GERD": (20, 75), "stable_angina": (45, 80), "pericarditis": (18, 60), "panic_attack": (18, 55)}

# per-dx defaults for slots DDXPlus does not have (manual, clinically typical); overridable by RNG variants
TEMPLATES = {
    "GERD": {"functional_impact": "none", "pattern": "episodic", "duration": "weeks", "trigger": {"postprandial": True, "exertional": False, "lying_down": True, "breathing": False, "movement": False},
             "relief": {"antacid": True, "rest": False}, "associated_default": {"acid_taste": True},
             "prior_test": ("ECG 6 months ago at a walk-in clinic: normal; told it was probably the stomach", "record"),
             "family": ("cad", False), "lifestyle": {"smoking": "occasional", "alcohol": "occasional"}, "history": {"reflux": True, "hypertension": False, "cad": False}, "medication": "antacid",
             "exam": {"Chest": "No reproducible tenderness; lungs clear", "Abdomen": "Mild epigastric tenderness"}, "tests": {"ECG": "Normal sinus rhythm", "Troponin": "Negative", "Endoscopy": "Grade A esophagitis"}},
    "stable_angina": {"functional_impact": "limits_activity", "pattern": "episodic", "duration": "months", "trigger": {"postprandial": False, "exertional": True, "lying_down": False, "breathing": False, "movement": False, "stress": True},
             "relief": {"rest": True, "antacid": False}, "associated_default": {},
             "prior_test": ("ECG 1 month ago: minor ST changes, cardiology follow-up advised", "record"),
             "family": ("cad", True), "lifestyle": {"smoking": "20_per_day", "alcohol": "occasional"}, "history": {"hypertension": True, "hyperlipidemia": True, "diabetes": False, "reflux": False}, "medication": "antihypertensive",
             "exam": {"Chest": "No chest wall tenderness; heart sounds normal"}, "tests": {"ECG": "Sinus rhythm, ST depression V4-V6", "Troponin": "Negative", "Stress_Test": "Positive at moderate workload"}},
    "pericarditis": {"functional_impact": "limits_activity", "pattern": "constant", "duration": "days", "trigger": {"postprandial": False, "exertional": False, "lying_down": True, "breathing": True, "movement": False},
             "relief": {"leaning_forward": True, "antacid": False, "rest": False}, "associated_default": {"fever": True},
             "prior_test": ("Chest X-ray last week for a cold: normal", "record"),
             "family": ("cad", False), "lifestyle": {"smoking": "none", "alcohol": "occasional"}, "history": {"reflux": False, "hypertension": False, "anxiety": False}, "medication": "painkiller",
             "exam": {"Chest": "Pericardial friction rub at the left sternal border", "Vital_Signs": {"Temperature": "37.9 C"}}, "tests": {"ECG": "Diffuse ST elevation with PR depression", "Troponin": "Mildly elevated", "Echo": "Small pericardial effusion"}},
    "panic_attack": {"functional_impact": "limits_activity", "pattern": "episodic", "duration": "weeks", "trigger": {"postprandial": False, "exertional": False, "lying_down": False, "breathing": False, "movement": False, "stress": True},
             "relief": {"rest": True, "antacid": False}, "associated_default": {"palpitations": True, "dyspnea": True, "dizziness": True},
             "prior_test": ("ECG in the emergency department 2 weeks ago: normal; sent home", "record"),
             "family": ("cad", False), "lifestyle": {"smoking": "10_per_day", "alcohol": "occasional"}, "history": {"anxiety": True, "hypertension": False, "reflux": False, "cad": False}, "medication": "anxiolytic",
             "exam": {"Chest": "No tenderness; lungs clear", "Vital_Signs": {"Heart_Rate": "98 bpm"}}, "tests": {"ECG": "Sinus tachycardia, otherwise normal", "Troponin": "Negative"}},
}
OCCUPATIONS = ["office clerk", "shop owner", "taxi driver", "nurse", "farmer", "software engineer", "retired", "teacher", "cook", "electrician"]


def load_ddxplus():
    ev = json.load(open(DDX / "release_evidences.json")); co = json.load(open(DDX / "release_conditions.json"))
    eng2fr = {v["cond-name-eng"]: k for k, v in co.items()}
    return ev, co, eng2fr


_DX_FOR = {}


def dx_for(condition_eng: str) -> str:
    return _DX_FOR.get(condition_eng, condition_eng)


def patients_for(condition_eng: str, n: int, rng: random.Random) -> list[dict]:
    rows = []
    lo, hi = AGE_RANGE.get(dx_for(condition_eng), (18, 85))
    with open(DDX / "release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if row["PATHOLOGY"] == condition_eng and lo <= int(row["AGE"]) <= hi:
                rows.append(row)
    rng.shuffle(rows)
    return rows[:n]


def map_patient(row: dict, dx: str, cfg, idx: int, rng: random.Random) -> Case:
    evs = ast.literal_eval(row["EVIDENCES"]); present = set(evs)
    slots = cfg.slots; T = TEMPLATES[dx]
    atoms = [{"id": "cp_present", "slot": "chief_complaint", "type": "bool", "true_value": True, "disclosure": "spontaneous"}]

    def cat_from(slot):
        sd = slots[slot]["ddxplus"]; ev = sd["evidence"]
        vals = [e.split("@_")[1] for e in evs if e.startswith(ev + "_@_")]
        mapped = [sd["map"][v] for v in vals if v in sd["map"]]
        return mapped

    loc = cat_from("location"); pref = LOC_PREF.get(dx, LOC_ORDER); loc_v = next((o for o in pref if o in loc), None)
    atoms.append({"id": "cp_location", "slot": "location", "type": "categorical", "true_value": loc_v or pref[0], "options": slots["location"]["options"], "disclosure": "on_general_ask"})
    q = cat_from("quality"); q_v = q[0] if q else rng.choice(slots["quality"]["options"])
    atoms.append({"id": "cp_quality", "slot": "quality", "type": "categorical", "true_value": q_v, "options": slots["quality"]["options"], "disclosure": "on_specific_ask"})
    inten = [int(e.split("@_")[1]) for e in evs if e.startswith("E_56_@_")]
    sev = "mild"
    if inten:
        i = inten[0]; sev = "mild" if i <= 3 else "moderate" if i <= 6 else "severe"
    if dx in ("GERD",) and sev == "severe":
        sev = "moderate"                        # DDXPlus intensities are inflated; keep GERD believable
    atoms.append({"id": "cp_severity", "slot": "severity", "type": "ordinal_3", "true_value": sev, "options": ["mild", "moderate", "severe"], "disclosure": "on_general_ask"})
    atoms.append({"id": "cp_functional", "slot": "functional_impact", "type": "categorical", "true_value": T["functional_impact"] if sev != "severe" else "wakes_at_night", "options": slots["functional_impact"]["options"], "disclosure": "on_specific_ask"})
    atoms.append({"id": "cp_pattern", "slot": "pattern", "type": "categorical", "true_value": T["pattern"], "options": ["episodic", "constant"], "disclosure": "on_specific_ask"})
    atoms.append({"id": "cp_duration", "slot": "duration", "type": "categorical", "true_value": T["duration"], "options": slots["duration"]["options"], "disclosure": "on_general_ask"})
    onset_v = [int(e.split("@_")[1]) for e in evs if e.startswith("E_59_@_")]
    atoms.append({"id": "cp_onset", "slot": "onset", "type": "categorical", "true_value": ("sudden" if onset_v and onset_v[0] >= 6 else "gradual"), "options": ["sudden", "gradual"], "disclosure": "on_specific_ask"})
    tagmap = slots["trigger"]["ddxplus"]["tags"]
    for tag in ["postprandial", "exertional", "lying_down", "breathing", "movement", "stress"]:
        code = tagmap.get(tag); v = (code in present) if code else None
        if v is None or (not v and tag in T["trigger"]):
            v = T["trigger"].get(tag, False)
        atoms.append({"id": f"trig_{tag}", "slot": "trigger", "type": "bool", "true_value": bool(v), "disclosure": "on_specific_ask", "tags": [tag]})
    for tag, v in T["relief"].items():
        atoms.append({"id": f"rel_{tag}", "slot": "relief", "type": "bool", "true_value": bool(v), "disclosure": "on_specific_ask", "tags": [tag]})
    rad = cat_from("radiation"); rad_v = next((o for o in ["arm", "jaw", "back"] if o in rad), "none")
    atoms.append({"id": "cp_radiation", "slot": "radiation", "type": "categorical", "true_value": rad_v, "options": slots["radiation"]["options"], "disclosure": "on_specific_ask"})
    atagmap = slots["associated"]["ddxplus"]["tags"]
    for tag in ["sweating", "dyspnea", "nausea", "palpitations", "dizziness", "acid_taste", "cough", "fever"]:
        code = atagmap.get(tag)
        v = (code in present) if code else T["associated_default"].get(tag, False)
        atoms.append({"id": f"assoc_{tag}", "slot": "associated", "type": "bool", "true_value": bool(v), "disclosure": "on_specific_ask", "tags": [tag]})
    ps = {"GERD": "mild", "stable_angina": "moderate", "pericarditis": "moderate", "panic_attack": "mild"}[dx]
    atoms.append({"id": "prior_severity", "slot": "prior_severity", "type": "ordinal_3", "true_value": ps, "options": ["mild", "moderate", "severe"], "disclosure": "on_specific_ask", "verifiable_by": "record"})
    pt, by = T["prior_test"]
    atoms.append({"id": "prior_test", "slot": "prior_test", "type": "text", "true_value": pt, "disclosure": "on_specific_ask", "verifiable_by": by})
    ftag, fv = T["family"]
    atoms.append({"id": f"fam_{ftag}", "slot": "family_history", "type": "bool", "true_value": bool(fv), "disclosure": "on_specific_ask", "verifiable_by": "family", "tags": [ftag]})
    smoke = "20_per_day" if "E_79" in present else T["lifestyle"]["smoking"]
    alc = "daily" if "E_78" in present else T["lifestyle"]["alcohol"]
    atoms.append({"id": "life_smoking", "slot": "lifestyle", "type": "categorical", "true_value": smoke, "options": slots["lifestyle"]["tag_values"]["smoking"], "disclosure": "on_specific_ask", "tags": ["smoking"]})
    atoms.append({"id": "life_alcohol", "slot": "lifestyle", "type": "categorical", "true_value": alc, "options": slots["lifestyle"]["tag_values"]["alcohol"], "disclosure": "on_specific_ask", "tags": ["alcohol"]})
    htagmap = slots["medical_history"]["ddxplus"]["tags"]
    hist = dict(T["history"])
    for tag, code in htagmap.items():
        if code in present:
            hist[tag] = True
    for tag, v in hist.items():
        atoms.append({"id": f"hx_{tag}", "slot": "medical_history", "type": "bool", "true_value": bool(v), "disclosure": "on_specific_ask", "tags": [tag]})
    atoms.append({"id": "meds", "slot": "medication", "type": "categorical", "true_value": T["medication"], "options": slots["medication"]["options"], "disclosure": "on_specific_ask"})
    d = {"case_id": f"cp_{idx:03d}", "cluster": cfg.name,
         "source": {"dataset": "ddxplus", "id": f"test:{row.get('_row')}", "condition": row["PATHOLOGY"], "reviewed_by": "pending", "reviewed_on": None,
                    "notes": "atoms from DDXPlus evidences where mapped; other slots from per-dx template"},
         "demographics": {"age": int(row["AGE"]), "sex": row["SEX"], "occupation": ("student" if int(row["AGE"]) < 22 else "retired" if int(row["AGE"]) >= 65 else rng.choice([o for o in OCCUPATIONS if o != "retired"]))},
         "labels": {"primary_dx": dx, "ddx_set": list(cfg.ddx_set), "red_flags": list(cfg.red_flags)},
         "atoms": atoms,
         "exam_and_tests": {"Physical_Examination_Findings": {"Vital_Signs": {"Blood_Pressure": f"{rng.randint(115, 150)}/{rng.randint(70, 92)} mmHg", "Heart_Rate": f"{rng.randint(64, 92)} bpm", "Temperature": "36.7 C"}, **T["exam"]},
                            "Test_Results": T["tests"]}}
    return Case.from_dict(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="chest_pain"); ap.add_argument("--per-dx", type=int, default=2); ap.add_argument("--start", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dx", default=None)
    args = ap.parse_args()
    cfg = load_cluster_config(args.cluster); rng = random.Random(args.seed)
    ev, co, eng2fr = load_ddxplus()
    out_dir = settings.clusters_dir / args.cluster / "cases"
    idx = args.start; written = []
    cond_map = cfg.raw.get("ddxplus_conditions", {})
    _DX_FOR.update({v: k for k, v in cond_map.items() if v})
    for dx, cond in cond_map.items():
        if not cond or (args.dx and dx != args.dx):
            continue
        rows = patients_for(cond, args.per_dx * 3, rng)
        # prefer patients whose pain is in the chest (location mapped) and who have an intensity value
        rows = sorted(rows, key=lambda r: -sum(1 for e in ast.literal_eval(r["EVIDENCES"]) if e.startswith("E_55_@_")))[: args.per_dx]
        for k, row in enumerate(rows):
            row["_row"] = k
            case = map_patient(row, dx, cfg, idx, rng)
            errs = validate(case, cfg)
            if errs:
                raise SystemExit(f"{case.case_id}: {errs}")
            (out_dir / f"{case.case_id}.json").write_text(json.dumps(case.to_dict(), indent=2, ensure_ascii=False))
            written.append((case.case_id, dx, case.demographics, [a.true_value for a in case.atoms if a.slot in ("location", "quality", "severity")]))
            idx += 1
    for w in written:
        print(w)


if __name__ == "__main__":
    main()
