# Session B: cp_006 (stable_angina) | channel vague:p=0.8 | seed 0

Doctor: manual, one question typed per turn. Patient: qwen3.7-plus phrasing, qwen3.7-max classifier, regex checker.

**Brief given to doctor:** 46-year-old man, office clerk, presenting with chest pain.

**Patient (opening):** I've been having chest pain.

**Doctor (t1):** Where exactly do you feel the pain?

**Patient:** It's been going on for several months. It's somewhere in the middle of my chest, kind of near the stomach. It's hard to ignore.

<sub>form=open hit=['cp_location'] disclosed=[cp_location=vague_center (true retrosternal), cp_severity=moderate, cp_duration=months] withheld=[] checker=ok retries=0 | oracle z: stable_angina 0.52, GERD 0.26, panic_attack 0.13 | u: vague_p08 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>

**Doctor (t2):** Let me be more specific: is it right behind the breastbone, just above the stomach, or over the left side of the chest?

**Patient:** It's right behind the breastbone.

<sub>form=forced_choice hit=['cp_location'] disclosed=[cp_location=retrosternal] withheld=[] checker=ok retries=0 | oracle z: stable_angina 0.68, GERD 0.19, panic_attack 0.05 | u: vague_p08 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>

**Doctor (t3):** What does the pain feel like, and what usually brings it on?

**Patient:** It's hard to describe, just uncomfortable. It gets worse when I exert myself or feel stressed.

<sub>form=open hit=['cp_quality', 'trig_postprandial', 'trig_exertional', 'trig_lying_down', 'trig_breathing', 'trig_movement', 'trig_stress'] disclosed=[cp_quality=vague_quality (true pressure), trig_postprandial=False, trig_exertional=True, trig_lying_down=False, trig_breathing=False, trig_movement=False, trig_stress=True] withheld=[] checker=ok retries=0 | oracle z: stable_angina 0.97, panic_attack 0.03, GERD 0.00 | u: vague_p08 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>

**Doctor (t4):** Does it ease off when you rest? And has anyone in your family had heart problems, or have you had an ECG before?

**Patient:** Yes, resting helps. My father had heart trouble, so it runs in the family. About a month ago, they did an ECG and found some minor changes, telling me to see a specialist.

<sub>form=yes_no hit=['rel_rest', 'fam_cad', 'prior_test'] disclosed=[rel_rest=True, fam_cad=True, prior_test=ECG 1 month ago: minor ST changes, cardiology follow-up advised] withheld=[] checker=ok retries=0 | oracle z: stable_angina 0.99, panic_attack 0.01, GERD 0.00 | u: vague_p08 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>

**Doctor (t5):** DIAGNOSIS READY: stable_angina
READOUT: {"topk": {"stable_angina": 0.7, "GERD": 0.15, "panic_attack": 0.05, "pericarditis": 0.05, "costochondritis": 0.05}, "confidence": 0.7}

**End:** stable_angina

<sub>form=None hit=[] disclosed=[] withheld=[] checker=ok retries=0 | oracle z: stable_angina 0.99, panic_attack 0.01, GERD 0.00 | u: vague_p08 1.00, cooperative 0.00, exaggerate_k1 0.00</sub>


**Episode ended.** LLM calls: conversational 5, meta 4 (cache hits 0).
