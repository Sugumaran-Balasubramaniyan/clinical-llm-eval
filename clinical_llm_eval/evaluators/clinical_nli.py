"""Clinical Natural Language Inference (NLI) and Contradiction Evaluator.

Evaluates clinical factual consistency and internal self-contradictions between
patient premise/vignette and the generated clinical response.
"""

from __future__ import annotations

import re
from typing import Any


class ClinicalNLIEvaluator:
    """Evaluates clinical consistency and detects contraindications between premise and response."""

    LABEL_ENTAILMENT = "ENTAILMENT"
    LABEL_NEUTRAL = "NEUTRAL"
    LABEL_CONTRADICTION = "CONTRADICTION"

    # Negation and avoidance cues within response sentences
    NEGATION_PATTERNS = [
        r'\b(?:avoid|avoiding|avoids|avoided)\b',
        r'\b(?:do\s+not|don\'?t)\s+(?:prescribe|give|use|administer|take|recommend|start|order)\b',
        r'\b(?:should\s+not|must\s+not|cannot|can\'?t)\s+(?:receive|be\s+given|take|be\s+used|be\s+prescribed)\b',
        r'\bnever\s+(?:give|prescribe|administer|use|take)\b',
        r'\b(?:contraindicated|contraindication|contraindications)\b',
        r'\b(?:unsafe|harmful|dangerous|not\s+recommended|inappropriate|strictly\s+avoided)\b',
        r'\b(?:hold|held|withhold|withheld|stop|stopped|discontinue|discontinued|cease)\b',
        r'\b(?:allergic\s+to|allergy\s+to|adverse\s+reaction\s+to)\b',
        r'\b(?:instead\s+of|alternative\s+to|substitute\s+for|in\s+place\s+of)\b',
        r'\b(?:caution\s+with|caution\s+against|warn\s+against|risk\s+of|toxicity\s+from)\b',
        r'\bwithout\b',
        r'\bfree\s+of\b',
    ]

    # --- 1. Renal Impairment Rules ---
    RENAL_PREMISE_PATTERNS = [
        r'\b(?:chronic\s+kidney\s+disease|ckd(?:\s+stage\s+[1-5])?|end[- ]stage\s+renal\s+disease|esrd|renal\s+failure|renal\s+impairment|kidney\s+failure|acute\s+kidney\s+injury|aki|acute\s+renal\s+failure|renal\s+insufficiency|nephrotic\s+syndrome|nephropathy)\b',
        r'\b(?:creatinine|cr|scr)\s*(?:level|is|=|:|\s+of)?\s*(?:[1-9]\d*(?:\.\d+)?)\s*(?:mg\/dl|mg%|\b)',
        r'\b(?:elevated|high|worsening|severe|marked|rising|abnormal)\s+(?:serum\s+)?creatinine\b',
        r'\bcreatinine\s+(?:is\s+)?(?:elevated|high|increased|severely\s+elevated)\b',
        r'\b(?:egfr|gfr)\s*(?:<|<=|of|is|=|:)?\s*(\d+)\s*(?:ml\/min(?:\/1\.73\s*m\^?2)?|\b)',
        r'\b(?:low|reduced|decreased|poor|severely\s+reduced)\s+(?:egfr|gfr|renal\s+function|kidney\s+function)\b',
        r'\b(?:hemodialysis|dialysis|peritoneal\s+dialysis|anuria|anuric|oliguria|oliguric)\b',
    ]

    NEPHROTOXIC_NSAIDS = [
        "ibuprofen", "naproxen", "ketorolac", "toradol", "indomethacin",
        "diclofenac", "meloxicam", "celecoxib", "advil", "motrin", "aleve",
        "nsaid", "nsaids", "high-dose nsaid", "high-dose nsaids",
    ]

    # --- 2. Hepatic Impairment Rules ---
    HEPATIC_PREMISE_PATTERNS = [
        r'\b(?:cirrhosis|decompensated\s+cirrhosis|acute\s+liver\s+failure|alf|hepatic\s+failure|severe\s+liver\s+disease|end[- ]stage\s+liver\s+disease|esld|hepatic\s+encephalopathy|child[- ]pugh\s+(?:class\s+)?[bc]|fulminant\s+hepatic\s+failure|severe\s+hepatitis|acute\s+liver\s+injury|jaundice\s+and\s+coagulopathy|markedly\s+elevated\s+lfts|liver\s+failure)\b',
    ]

    HEPATOTOXINS = [
        "methotrexate", "valproic acid", "valproate", "depakote", "amiodarone",
        "ketoconazole", "isoniazid",
    ]

    # --- 3. Allergy Rules ---
    PENICILLIN_ALLERGY_PATTERNS = [
        r'\b(?:allergic\s+to\s+(?:penicillins?|pcn|amoxicillin|ampicillin|augmentin|beta[- ]lactams?)|penicillin\s+allergy|pcn\s+allergy|allergy\s*:\s*penicillin|amoxicillin\s+allergy|history\s+of\s+penicillin\s+(?:allergy|anaphylaxis|hives|reaction)|severe\s+penicillin\s+allergy)\b',
    ]

    PENICILLIN_DRUGS = [
        "penicillin", "penicillin vk", "penicillin g", "amoxicillin", "amoxicillin-clavulanate",
        "amoxicillin/clavulanic acid", "amoxicillin/clavulanate", "augmentin", "ampicillin",
        "ampicillin-sulbactam", "ampicillin/sulbactam", "unasyn", "piperacillin",
        "piperacillin-tazobactam", "piperacillin/tazobactam", "zosyn", "oxacillin",
        "nafcillin", "dicloxacillin", "methicillin", "ticarcillin",
    ]

    CROSS_REACTIVE_CEPHALOSPORINS = [
        "cephalexin", "keflex", "cefazolin", "ancef", "cefaclor", "cefuroxime",
        "ceftriaxone", "cefepime", "cefdinir", "cefotaxime",
    ]

    SULFA_ALLERGY_PATTERNS = [
        r'\b(?:allergic\s+to\s+(?:sulfa|sulfonamides?|bactrim|septra|tmp-smx)|sulfa\s+allergy|sulfonamide\s+allergy|allergy\s*:\s*sulfa|bactrim\s+allergy|history\s+of\s+sulfa\s+allergy)\b',
    ]

    SULFA_DRUGS = [
        "bactrim", "septra", "tmp-smx", "trimethoprim-sulfamethoxazole",
        "trimethoprim/sulfamethoxazole", "sulfamethoxazole", "sulfadiazine",
        "sulfasalazine", "zonisamide", "sulfisoxazole",
    ]

    ASPIRIN_ALLERGY_PATTERNS = [
        r'\b(?:allergic\s+to\s+(?:aspirin|nsaids?|ibuprofen)|aspirin\s+allergy|nsaid\s+allergy|allergy\s*:\s*(?:aspirin|nsaid)|samter\'?s\s+triad|aspirin-exacerbated\s+respiratory\s+disease|aerd)\b',
    ]

    ASPIRIN_NSAIDS = [
        "aspirin", "acetylsalicylic acid", "bayer", "ibuprofen", "naproxen",
        "ketorolac", "toradol", "indomethacin", "diclofenac", "meloxicam", "celecoxib",
    ]

    OPIOID_ALLERGY_PATTERNS = [
        r'\b(?:allergic\s+to\s+(?:codeine|morphine|opioids?)|codeine\s+allergy|morphine\s+allergy|opioid\s+allergy)\b',
    ]

    OPIOID_DRUGS = [
        "codeine", "morphine", "tramadol", "oxycodone", "hydrocodone",
        "hydromorphone", "fentanyl",
    ]

    # --- 4. Bleeding Risk Rules ---
    BLEEDING_PREMISE_PATTERNS = [
        r'\b(?:active\s+gi\s+bleed(?:ing)?|upper\s+gi\s+bleed(?:ing)?|lower\s+gi\s+bleed(?:ing)?|melena|hematochezia|hematemesis|intracranial\s+hemorrhage|ich|hemorrhagic\s+stroke|bleeding\s+peptic\s+ulcer|subdural\s+hematoma|epidural\s+hematoma|subarachnoid\s+hemorrhage|active\s+hemorrhage|active\s+bleeding|severe\s+thrombocytopenia|massive\s+hemoptysis|retroperitoneal\s+bleed(?:ing)?)\b',
        r'\b(?:platelet\s+count|platelets?)\s*(?:<|<=|of|is|=|:)?\s*([0-4]?[0-9])\s*(?:k|\*10\^3|\*10\^9|\/ul|\/microliter|\b)',
    ]

    ANTICOAGULANTS_ANTIPLATELETS = [
        "heparin", "unfractionated heparin", "ufh", "heparin bolus", "heparin drip",
        "enoxaparin", "lovenox", "dalteparin", "fragmin", "fondaparinux", "arixtra",
        "warfarin", "coumadin", "apixaban", "eliquis", "rivaroxaban", "xarelto",
        "dabigatran", "pradaxa", "edoxaban", "savaysa", "bivalirudin", "argatroban",
        "aspirin", "clopidogrel", "plavix", "ticagrelor", "brilinta", "prasugrel",
        "effient", "dipyridamole", "tpa", "alteplase", "tenecteplase", "streptokinase",
    ]

    # --- 5. Asthma / Bronchospasm Rules ---
    ASTHMA_PREMISE_PATTERNS = [
        r'\b(?:severe\s+asthma|acute\s+asthma\s+exacerbation|status\s+asthmaticus|severe\s+bronchospasm|active\s+wheezing|reactive\s+airway\s+disease|uncontrolled\s+asthma|brittle\s+asthma|history\s+of\s+severe\s+asthma)\b',
    ]

    NON_SELECTIVE_BETA_BLOCKERS = [
        "propranolol", "inderal", "nadolol", "corgard", "timolol", "sotalol",
        "pindolol", "carvedilol", "labetalol", "non-selective beta-blocker",
        "non-selective beta-blockers",
    ]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences/clauses for granular context analysis."""
        if not text:
            return []
        raw_parts = re.split(r'[\n\.\;\!\?]+', text)
        return [part.strip() for part in raw_parts if part.strip()]

    def _is_negated_or_avoided(self, sentence: str, term: str) -> bool:
        """Check if a drug/intervention mention in a sentence is negated, avoided, or warned against."""
        sentence_lower = sentence.lower()
        term_lower = term.lower()
        if term_lower not in sentence_lower:
            return False

        for neg_pat in self.NEGATION_PATTERNS:
            if re.search(neg_pat, sentence_lower):
                return True
        return False

    def _has_affirmative_recommendation(self, response: str, drug_terms: list[str]) -> tuple[bool, str | None]:
        """Check if response contains an un-negated positive recommendation for any of the drug terms."""
        sentences = self._split_sentences(response)
        for sentence in sentences:
            for drug in drug_terms:
                pattern = rf'\b{re.escape(drug)}\b'
                if re.search(pattern, sentence, re.IGNORECASE):
                    if not self._is_negated_or_avoided(sentence, drug):
                        return True, drug
        return False, None

    def _check_renal_contradictions(self, prompt: str, response: str) -> list[str]:
        """Detect renal function contradictions (e.g. NSAIDs, Metformin, IV contrast in renal failure)."""
        contradictions: list[str] = []
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        # Detect renal impairment in prompt
        has_renal_impairment = False
        renal_match_str = "renal impairment"

        for pat in self.RENAL_PREMISE_PATTERNS:
            m = re.search(pat, prompt_lower)
            if m:
                if "cr" in pat or "creatinine" in pat:
                    cr_match = re.search(r'\b(?:creatinine|cr|scr)\s*(?:level|is|=|:|\s+of)?\s*([0-9]+(?:\.[0-9]+)?)\b', prompt_lower)
                    if cr_match:
                        try:
                            val = float(cr_match.group(1))
                            if val >= 1.5:
                                has_renal_impairment = True
                                renal_match_str = f"creatinine {val} mg/dL"
                                break
                        except ValueError:
                            pass
                elif "gfr" in pat:
                    gfr_match = re.search(r'\b(?:egfr|gfr)\s*(?:<|<=|of|is|=|:)?\s*([0-9]+)\b', prompt_lower)
                    if gfr_match:
                        try:
                            val = int(gfr_match.group(1))
                            if val <= 45:
                                has_renal_impairment = True
                                renal_match_str = f"eGFR {val} mL/min"
                                break
                        except ValueError:
                            pass
                else:
                    has_renal_impairment = True
                    renal_match_str = m.group(0)
                    break

        if not has_renal_impairment:
            return contradictions

        # 1. Nephrotoxic NSAID prescription
        has_nsaid, culprit_nsaid = self._has_affirmative_recommendation(response, self.NEPHROTOXIC_NSAIDS)
        if has_nsaid and culprit_nsaid:
            contradictions.append(
                f"Renal Contradiction: Patient has documented renal impairment ({renal_match_str}), "
                f"but response recommends nephrotoxic NSAID ({culprit_nsaid})."
            )

        # 2. Metformin without adjustment / in severe renal failure
        has_metformin, culprit_met = self._has_affirmative_recommendation(response, ["metformin", "glucophage"])
        if has_metformin and culprit_met:
            is_severe_renal = any(term in prompt_lower for term in ["ckd stage 4", "ckd stage 5", "esrd", "end-stage", "dialysis", "cr 3", "cr 4", "cr 5", "creatinine 3", "creatinine 4", "creatinine 5", "egfr 1", "egfr 2"])
            cr_num = re.search(r'\b(?:creatinine|cr|scr)\s*(?:level|is|=|:|\s+of)?\s*([0-9]+(?:\.[0-9]+)?)\b', prompt_lower)
            if cr_num:
                try:
                    if float(cr_num.group(1)) >= 2.0:
                        is_severe_renal = True
                except ValueError:
                    pass

            if is_severe_renal:
                has_dose_adjust = any(term in response_lower for term in ["dose adjustment", "reduce dose", "renal dosing", "contraindicated", "monitor egfr", "hold metformin"])
                if not has_dose_adjust:
                    contradictions.append(
                        f"Renal Contradiction: Patient has severe renal impairment ({renal_match_str}), "
                        f"but response prescribes Metformin without dose adjustment / contraindication assessment (risk of lactic acidosis)."
                    )

        # 3. IV contrast without hydration
        has_contrast = bool(re.search(r'\b(?:iv\s+contrast|iodinated\s+contrast|contrast-enhanced\s+ct|ct\s+with\s+contrast|contrast\s+ct)\b', response_lower))
        if has_contrast:
            is_negated = any(re.search(neg, response_lower) for neg in self.NEGATION_PATTERNS)
            has_hydration = any(term in response_lower for term in ["hydration", "saline", "iv fluids", "pre-hydrate", "normal saline", "bicarbonate", "hydrate"])
            if not is_negated and not has_hydration:
                contradictions.append(
                    f"Renal Contradiction: Patient has renal impairment ({renal_match_str}), "
                    "but response recommends IV contrast without necessary hydration or renal protection protocols."
                )

        return contradictions

    def _check_hepatic_contradictions(self, prompt: str, response: str) -> list[str]:
        """Detect hepatic impairment contradictions (e.g. acetaminophen, hepatotoxins in cirrhosis/ALF)."""
        contradictions: list[str] = []
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        has_hepatic_impairment = False
        hepatic_match_str = "hepatic impairment"

        for pat in self.HEPATIC_PREMISE_PATTERNS:
            m = re.search(pat, prompt_lower)
            if m:
                has_hepatic_impairment = True
                hepatic_match_str = m.group(0)
                break

        if not has_hepatic_impairment:
            return contradictions

        # Check Acetaminophen / Paracetamol
        has_apap, culprit_apap = self._has_affirmative_recommendation(
            response, ["acetaminophen", "paracetamol", "tylenol", "apap"]
        )
        if has_apap and culprit_apap:
            is_acute_failure = any(term in prompt_lower for term in ["acute liver failure", "alf", "fulminant", "decompensated cirrhosis", "hepatic encephalopathy"])
            has_high_dose = bool(re.search(r'\b(?:3000\s*mg|4000\s*mg|4\s*g(?:rams?)?|6\s*g(?:rams?)?|1000\s*mg\s+(?:every|q)\s*(?:4|6)\s*h|high-dose)\b', response_lower))
            if is_acute_failure or has_high_dose:
                contradictions.append(
                    f"Hepatic Contradiction: Patient has severe hepatic impairment ({hepatic_match_str}), "
                    f"but response recommends hepatotoxic / high-dose acetaminophen ({culprit_apap})."
                )

        # Check other hepatotoxins
        has_hepatotoxin, culprit_tox = self._has_affirmative_recommendation(response, self.HEPATOTOXINS)
        if has_hepatotoxin and culprit_tox:
            contradictions.append(
                f"Hepatic Contradiction: Patient has hepatic impairment ({hepatic_match_str}), "
                f"but response recommends hepatotoxic agent ({culprit_tox}) contraindicated in liver failure."
            )

        return contradictions

    def _check_allergy_contradictions(self, prompt: str, response: str) -> list[str]:
        """Detect documented drug allergy contradictions (e.g. Penicillin, Sulfa, Aspirin, Opioid)."""
        contradictions: list[str] = []
        prompt_lower = prompt.lower()

        # 1. Penicillin / Beta-lactam Allergy
        has_pcn_allergy = any(re.search(pat, prompt_lower) for pat in self.PENICILLIN_ALLERGY_PATTERNS)
        if has_pcn_allergy:
            has_pcn, culprit_pcn = self._has_affirmative_recommendation(response, self.PENICILLIN_DRUGS)
            if has_pcn and culprit_pcn:
                contradictions.append(
                    f"Allergy Contradiction: Patient has documented penicillin allergy, "
                    f"but response prescribes allergen or cross-reactive drug ({culprit_pcn})."
                )
            else:
                is_anaphylactic = any(term in prompt_lower for term in ["anaphylaxis", "hives", "angioedema", "severe allergy", "respiratory distress"])
                if is_anaphylactic:
                    has_ceph, culprit_ceph = self._has_affirmative_recommendation(response, self.CROSS_REACTIVE_CEPHALOSPORINS)
                    if has_ceph and culprit_ceph:
                        contradictions.append(
                            f"Allergy Contradiction: Patient has documented severe/anaphylactic penicillin allergy, "
                            f"but response prescribes cross-reactive cephalosporin ({culprit_ceph})."
                        )

        # 2. Sulfa Allergy
        has_sulfa_allergy = any(re.search(pat, prompt_lower) for pat in self.SULFA_ALLERGY_PATTERNS)
        if has_sulfa_allergy:
            has_sulfa, culprit_sulfa = self._has_affirmative_recommendation(response, self.SULFA_DRUGS)
            if has_sulfa and culprit_sulfa:
                contradictions.append(
                    f"Allergy Contradiction: Patient has documented sulfa allergy, "
                    f"but response prescribes sulfonamide medication ({culprit_sulfa})."
                )

        # 3. Aspirin / NSAID Allergy
        has_aspirin_allergy = any(re.search(pat, prompt_lower) for pat in self.ASPIRIN_ALLERGY_PATTERNS)
        if has_aspirin_allergy:
            has_asp, culprit_asp = self._has_affirmative_recommendation(response, self.ASPIRIN_NSAIDS)
            if has_asp and culprit_asp:
                contradictions.append(
                    f"Allergy Contradiction: Patient has documented aspirin / NSAID allergy, "
                    f"but response prescribes allergen ({culprit_asp})."
                )

        # 4. Opioid Allergy
        has_opioid_allergy = any(re.search(pat, prompt_lower) for pat in self.OPIOID_ALLERGY_PATTERNS)
        if has_opioid_allergy:
            has_op, culprit_op = self._has_affirmative_recommendation(response, self.OPIOID_DRUGS)
            if has_op and culprit_op:
                contradictions.append(
                    f"Allergy Contradiction: Patient has documented opioid allergy, "
                    f"but response prescribes opioid medication ({culprit_op})."
                )

        return contradictions

    def _check_bleeding_contradictions(self, prompt: str, response: str) -> list[str]:
        """Detect bleeding risk contradictions (e.g. anticoagulation/antiplatelets during active GI bleed/ICH)."""
        contradictions: list[str] = []
        prompt_lower = prompt.lower()

        has_bleeding = False
        bleed_match_str = "active bleeding / hemorrhage"

        for pat in self.BLEEDING_PREMISE_PATTERNS:
            m = re.search(pat, prompt_lower)
            if m:
                has_bleeding = True
                bleed_match_str = m.group(0)
                break

        if not has_bleeding:
            return contradictions

        has_anticoag, culprit_anticoag = self._has_affirmative_recommendation(
            response, self.ANTICOAGULANTS_ANTIPLATELETS
        )
        if has_anticoag and culprit_anticoag:
            contradictions.append(
                f"Bleeding Risk Contradiction: Patient has active bleeding / intracranial hemorrhage ({bleed_match_str}), "
                f"but response recommends anticoagulation / antiplatelet therapy ({culprit_anticoag})."
            )

        return contradictions

    def _check_bronchospasm_contradictions(self, prompt: str, response: str) -> list[str]:
        """Detect asthma / bronchospasm contradictions (e.g. non-selective beta-blockers in severe asthma)."""
        contradictions: list[str] = []
        prompt_lower = prompt.lower()

        has_asthma = False
        asthma_match_str = "severe asthma / bronchospasm"

        for pat in self.ASTHMA_PREMISE_PATTERNS:
            m = re.search(pat, prompt_lower)
            if m:
                has_asthma = True
                asthma_match_str = m.group(0)
                break

        if not has_asthma:
            return contradictions

        has_bb, culprit_bb = self._has_affirmative_recommendation(
            response, self.NON_SELECTIVE_BETA_BLOCKERS
        )
        if has_bb and culprit_bb:
            contradictions.append(
                f"Bronchospasm Contradiction: Patient has severe asthma / bronchospasm ({asthma_match_str}), "
                f"but response recommends non-selective beta-blocker ({culprit_bb}) which can trigger severe bronchoconstriction."
            )

        return contradictions

    def _determine_nli_label(self, has_contradiction: bool, prompt: str, response: str) -> str:
        """Determine whether clinical relationship is ENTAILMENT, NEUTRAL, or CONTRADICTION."""
        if has_contradiction:
            return self.LABEL_CONTRADICTION

        clean_resp = response.strip()
        if not clean_resp or len(clean_resp.split()) < 3:
            return self.LABEL_NEUTRAL

        return self.LABEL_ENTAILMENT

    def evaluate(self, prompt: str, response: str) -> dict[str, Any]:
        """Evaluate clinical factual consistency and internal self-contradictions.

        Args:
            prompt: Clinical vignette / patient premise containing history, labs, or allergies.
            response: Model-generated clinical recommendation / prescription.

        Returns:
            Dict containing:
                - has_contradiction: bool
                - contradictions: list[str]
                - consistency_score: float (1.0 = consistent, 0.0 = severe contradiction)
                - clinical_nli_label: 'ENTAILMENT' | 'NEUTRAL' | 'CONTRADICTION'
        """
        all_contradictions: list[str] = []

        all_contradictions.extend(self._check_renal_contradictions(prompt, response))
        all_contradictions.extend(self._check_hepatic_contradictions(prompt, response))
        all_contradictions.extend(self._check_allergy_contradictions(prompt, response))
        all_contradictions.extend(self._check_bleeding_contradictions(prompt, response))
        all_contradictions.extend(self._check_bronchospasm_contradictions(prompt, response))

        has_contradiction = len(all_contradictions) > 0
        nli_label = self._determine_nli_label(has_contradiction, prompt, response)

        if has_contradiction:
            consistency_score = 0.0
        elif nli_label == self.LABEL_NEUTRAL:
            consistency_score = 0.5
        else:
            consistency_score = 1.0

        return {
            "has_contradiction": has_contradiction,
            "contradictions": all_contradictions,
            "consistency_score": consistency_score,
            "clinical_nli_label": nli_label,
        }

    @classmethod
    def compute_batch_metrics(cls, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregated contradiction and consistency metrics across a batch of evaluation results.

        Args:
            results: List of evaluation result dicts from `evaluate()`.

        Returns:
            Dict containing:
                - contradiction_rate: float
                - mean_consistency_score: float
                - total_contradictions_detected: int
                - entailment_rate: float
                - total_evaluated: int
        """
        total_evaluated = len(results)
        if total_evaluated == 0:
            return {
                "contradiction_rate": 0.0,
                "mean_consistency_score": 1.0,
                "total_contradictions_detected": 0,
                "entailment_rate": 0.0,
                "total_evaluated": 0,
            }

        contradictions_count = sum(1 for r in results if bool(r.get("has_contradiction", False)))
        contradiction_rate = contradictions_count / total_evaluated

        entailment_count = sum(1 for r in results if r.get("clinical_nli_label") == cls.LABEL_ENTAILMENT)
        entailment_rate = entailment_count / total_evaluated

        total_contradictions = sum(len(r.get("contradictions", [])) for r in results)

        consistency_scores = [float(r.get("consistency_score", 0.0)) for r in results]
        mean_consistency = sum(consistency_scores) / total_evaluated

        return {
            "contradiction_rate": round(contradiction_rate, 4),
            "mean_consistency_score": round(mean_consistency, 4),
            "total_contradictions_detected": total_contradictions,
            "entailment_rate": round(entailment_rate, 4),
            "total_evaluated": total_evaluated,
        }
