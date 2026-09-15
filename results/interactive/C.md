# Session C: cp_010 (panic_attack) | channel self_dx:conviction=insist | seed 0

Doctor: manual, one question typed per turn. Patient: qwen3.7-plus phrasing, qwen3.7-max classifier, regex checker.

**Brief given to doctor:** 26-year-old man, taxi driver, presenting with chest pain.

**Patient (opening):** I'm pretty sure I have angina. I've been having chest pain.

**Doctor (t1):** What makes you think it's angina? Tell me about the pain — where is it and what does it feel like?

**Patient:** It's over the left side of my chest, near the heart, and feels like a heavy pressure or squeezing. It's bearable and has been going on for a couple of weeks. I'm sure it's angina, so please check me for that.

<sub>form=open hit=['cp_location', 'cp_quality'] disclosed=[cp_location=precordial, cp_quality=pressure, cp_severity=mild, cp_duration=weeks] withheld=[] checker=ok retries=0 | oracle z: costochondritis 0.38, stable_angina 0.27, panic_attack 0.26 | u: cooperative 0.49, omit_q05 0.49, vague_p08 0.02</sub>

**Doctor (t2):** What brings it on — exertion, stress, eating? And does it come with palpitations, sweating or feeling faint?

**Patient:** It doesn't change when I'm working or after I eat, but it does get worse when I'm stressed. I've also been having palpitations, sweating, and dizziness.

<sub>form=forced_choice hit=['trig_exertional', 'trig_stress', 'trig_postprandial', 'assoc_palpitations', 'assoc_sweating', 'assoc_dizziness'] disclosed=[trig_exertional=False, trig_stress=True, trig_postprandial=False, assoc_palpitations=True, assoc_sweating=True, assoc_dizziness=True] withheld=[] checker=ok retries=0 | oracle z: panic_attack 1.00, stable_angina 0.00, pericarditis 0.00 | u: cooperative 0.86, omit_q05 0.11, vague_p08 0.03</sub>

**Doctor (t3):** Has anyone in your family had heart disease? And have you had any tests done before, like an ECG?

**Patient:** I don't know of any heart trouble in my family. When I went to the emergency department two weeks ago, they did an ECG that came back normal and sent me home. But I still think it is angina, so please check me for that.

<sub>form=yes_no hit=['fam_cad', 'prior_test'] disclosed=[fam_cad=False, prior_test=ECG in the emergency department 2 weeks ago: normal; sent home] withheld=[] checker=ok retries=0 | oracle z: panic_attack 1.00, stable_angina 0.00, pericarditis 0.00 | u: cooperative 0.86, omit_q05 0.11, vague_p08 0.03</sub>

**Doctor (t4):** I understand you're worried about angina. Your ECG was normal and the pain comes with stress rather than exertion, which points away from the heart. Do you have any history of anxiety, and are you taking any medication?

**Patient:** Yes, I have anxiety. I take something for my nerves.

<sub>form=yes_no hit=['hx_anxiety', 'meds'] disclosed=[hx_anxiety=True, meds=anxiolytic] withheld=[] checker=ok retries=0 | oracle z: panic_attack 1.00, stable_angina 0.00, pericarditis 0.00 | u: cooperative 0.86, omit_q05 0.11, vague_p08 0.03</sub>

**Doctor (t5):** DIAGNOSIS READY: panic_attack
READOUT: {"topk": {"panic_attack": 0.75, "stable_angina": 0.1, "costochondritis": 0.1, "GERD": 0.03, "pericarditis": 0.02}, "confidence": 0.8}

**End:** panic_attack

<sub>form=None hit=[] disclosed=[] withheld=[] checker=ok retries=0 | oracle z: panic_attack 1.00, stable_angina 0.00, pericarditis 0.00 | u: cooperative 0.86, omit_q05 0.11, vague_p08 0.03</sub>


**Episode ended.** LLM calls: conversational 5, meta 4 (cache hits 0).
