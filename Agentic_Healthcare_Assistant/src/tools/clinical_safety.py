"""
Clinical Safety and Medication Contraindication Guard.
Evaluates clinical queries and proposed therapies against patient-specific
diagnoses and organ function to prevent adverse drug events (e.g. NSAIDs in CKD).
"""

import re
from typing import Any, Dict, List, Optional


class ClinicalSafetyGuard:
    """
    Rule-based clinical contraindication and drug safety monitoring engine.
    Cross-checks patient conditions (e.g. CKD, Hypertension, Diabetes) against
    medication queries to ensure evidence-based pharmacological safety.
    """

    CONTRAINDICATION_RULES = [
        {
            "condition_keywords": ["kidney", "renal", "ckd", "nephro", "dialysis"],
            "drug_keywords": [
                "ibuprofen", "advil", "motrin", "naproxen", "aleve",
                "diclofenac", "voltaren", "celecoxib", "celebrex",
                "meloxicam", "mobic", "ketorolac", "toradol", "nsaid", "nsaids"
            ],
            "severity": "HIGH",
            "warning_title": "CRITICAL CONTRAINDICATION: NSAIDs in Chronic Kidney Disease",
            "message": (
                "Non-Steroidal Anti-Inflammatory Drugs (NSAIDs) inhibit renal prostaglandins, "
                "inducing afferent arteriolar vasoconstriction, sharp drops in glomerular filtration rate (eGFR), "
                "and potential acute-on-chronic kidney injury."
            ),
            "safe_alternatives": (
                "For mild-to-moderate analgesia, **Acetaminophen (Paracetamol)** at doses up to 2 g/day is the "
                "preferred renal-safe alternative. For severe pain, consult treating nephrologist."
            )
        },
        {
            "condition_keywords": ["hypertension", "blood pressure", "htn"],
            "drug_keywords": ["pseudoephedrine", "sudafed", "phenylephrine", "decongestant"],
            "severity": "MODERATE",
            "warning_title": "PRECAUTION: Sympathomimetic Decongestants in Hypertension",
            "message": (
                "Oral decongestants stimulate alpha-adrenergic receptors, causing systemic vasoconstriction "
                "and precipitous elevations in blood pressure."
            ),
            "safe_alternatives": (
                "Consider saline nasal sprays or topical intranasal corticosteroids (e.g., Fluticasone) "
                "which have minimal systemic vascular effects."
            )
        },
        {
            "condition_keywords": ["diabetes", "diabetic", "glycemic"],
            "drug_keywords": ["prednisone", "dexamethasone", "methylprednisolone", "steroid", "corticosteroid"],
            "severity": "MODERATE",
            "warning_title": "MONITORING ALERT: Corticosteroid-Induced Hyperglycemia",
            "message": (
                "Systemic glucocorticoids increase hepatic gluconeogenesis and induce peripheral insulin resistance, "
                "frequently precipitating marked glycemic excursions."
            ),
            "safe_alternatives": (
                "If steroids are clinically required, implement intensified blood glucose surveillance "
                "and proactive insulin/hypoglycemic dose titration under endocrinology supervision."
            )
        }
    ]

    EMERGENCY_RED_FLAGS = [
        {
            "category": "Cardiac / Acute Coronary Syndrome",
            "keywords": [
                "chest pain", "crushing chest", "pressure in chest", "radiating to jaw",
                "radiating to left arm", "radiating to arm", "heart attack"
            ],
            "action": "🚨 CALL 911 / EMERGENCY MEDICAL DISPATCH IMMEDIATELY",
            "guidance": (
                "Severe central chest pressure or pain radiating to the jaw/arm indicates possible "
                "Acute Myocardial Infarction. Do not wait for an outpatient appointment—seek immediate emergency ER care."
            )
        },
        {
            "category": "Neurological / Acute Stroke (FAST)",
            "keywords": [
                "slurred speech", "facial droop", "face drooping", "arm weakness",
                "loss of speech", "sudden paralysis", "stroke", "unable to speak"
            ],
            "action": "🚨 CODE STROKE ALERT (CALL 911 / EMERGENCY SERVICES)",
            "guidance": (
                "Acute focal neurological deficit (FAST criteria: Face, Arms, Speech, Time) is a time-critical emergency. "
                "Immediate emergency room transport is vital for thrombolytic eligibility."
            )
        },
        {
            "category": "Respiratory Distress",
            "keywords": [
                "can't breathe", "cannot breathe", "severe shortness of breath",
                "struggling to breathe", "blue lips", "cyanosis", "gasping"
            ],
            "action": "🚨 EMERGENCY AIRWAY & OXYGENATION ALERT",
            "guidance": (
                "Severe acute dyspnea with impending respiratory failure requires urgent emergency department resuscitation "
                "and supplemental oxygenation."
            )
        },
        {
            "category": "Massive Hemorrhage / Hemoptysis",
            "keywords": [
                "coughing up blood", "vomiting blood", "massive bleeding",
                "uncontrolled bleeding", "passed out and bleeding"
            ],
            "action": "🚨 ACUTE HEMORRHAGIC SHOCK ALERT",
            "guidance": (
                "Active gastrointestinal hemorrhage or hemoptysis requires immediate emergency room evaluation, "
                "fluid resuscitation, and hemodynamic monitoring."
            )
        }
    ]

    DRUG_INTERACTION_RULES = [
        {
            "pair": ["ace_inhibitor_or_arb", "potassium_sparing"],
            "drugs_a": ["lisinopril", "ramipril", "enalapril", "losartan", "valsartan", "candesartan"],
            "drugs_b": ["spironolactone", "eplerenone", "triamterene", "amiloride", "potassium"],
            "severity": "CRITICAL",
            "title": "Severe Hyperkalemia Risk: ACEi/ARB + Potassium-Sparing Agent",
            "message": (
                "Concurrent administration of Renin-Angiotensin blockers and Potassium-sparing diuretics markedly impairs "
                "renal potassium excretion, risking life-threatening hyperkalemia and cardiac dysrhythmias."
            )
        },
        {
            "pair": ["anticoagulant", "nsaid"],
            "drugs_a": ["warfarin", "coumadin", "eliquis", "apixaban", "xarelto", "rivaroxaban", "pradaxa"],
            "drugs_b": ["aspirin", "ibuprofen", "naproxen", "diclofenac", "meloxicam", "ketorolac"],
            "severity": "HIGH",
            "title": "Severe Hemorrhagic Risk: Anticoagulant + NSAID",
            "message": (
                "Co-administration of direct oral anticoagulants or warfarin with NSAIDs damages gastric mucosal barriers "
                "and inhibits platelet function, drastically multiplying major gastrointestinal hemorrhage risks."
            )
        },
        {
            "pair": ["statin", "macrolide"],
            "drugs_a": ["atorvastatin", "simvastatin", "lovastatin"],
            "drugs_b": ["clarithromycin", "erythromycin"],
            "severity": "HIGH",
            "title": "Rhabdomyolysis Risk: CYP3A4-Metabolized Statin + Macrolide",
            "message": (
                "Potent CYP3A4 inhibition by macrolide antibiotics elevates systemic statin concentrations dramatically, "
                "escalating risks of acute myopathy and life-threatening rhabdomyolysis."
            )
        }
    ]

    @classmethod
    def check_safety(cls, query: str, conditions: List[str]) -> Optional[Dict[str, Any]]:
        """
        Scans query and patient conditions for contraindications.
        Returns safety alert payload if a contraindication is detected, otherwise None.
        """
        q_lower = query.lower()
        cond_lower = " ".join([c.lower() for c in conditions]) if conditions else ""

        for rule in cls.CONTRAINDICATION_RULES:
            # Check if condition matches patient
            cond_match = any(ck in cond_lower or ck in q_lower for ck in rule["condition_keywords"])
            if not cond_match:
                continue

            # Check if contraindicated drug is mentioned in query
            for dk in rule["drug_keywords"]:
                pattern = rf"\b{re.escape(dk)}\b"
                if re.search(pattern, q_lower):
                    return {
                        "has_warning": True,
                        "severity": rule["severity"],
                        "warning_title": rule["warning_title"],
                        "detected_substance": dk.capitalize(),
                        "message": rule["message"],
                        "safe_alternatives": rule["safe_alternatives"]
                    }

        return None

    @classmethod
    def check_emergency_red_flags(cls, query: str) -> Optional[Dict[str, Any]]:
        """
        Detects acute clinical emergency red flags requiring immediate ER dispatch.
        """
        q_lower = query.lower()
        for red_flag in cls.EMERGENCY_RED_FLAGS:
            for kw in red_flag["keywords"]:
                if re.search(rf"\b{re.escape(kw)}\b", q_lower):
                    return {
                        "is_emergency": True,
                        "category": red_flag["category"],
                        "detected_symptom": kw.title(),
                        "action_required": red_flag["action"],
                        "clinical_guidance": red_flag["guidance"]
                    }
        return None

    @classmethod
    def check_drug_drug_interactions(cls, medications: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluates a list of patient medications or proposed drugs for severe drug-drug interactions.
        """
        interactions = []
        meds_lower = [m.lower().strip() for m in medications if m]
        all_meds_text = " ".join(meds_lower)

        for rule in cls.DRUG_INTERACTION_RULES:
            has_drug_a = any(re.search(rf"\b{re.escape(da)}\b", all_meds_text) for da in rule["drugs_a"])
            has_drug_b = any(re.search(rf"\b{re.escape(db)}\b", all_meds_text) for db in rule["drugs_b"])

            if has_drug_a and has_drug_b:
                interactions.append({
                    "title": rule["title"],
                    "severity": rule["severity"],
                    "message": rule["message"]
                })

        return interactions

