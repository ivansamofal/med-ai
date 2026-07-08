"""One-off generator for the sample guideline PDFs used by the knowledge base.

These are short, deliberately simplified, synthetic documents for a portfolio
project — not real clinical guidance. Run again only if you want to
regenerate/replace the PDFs under data/knowledge_base/guidelines/.
"""

from pathlib import Path

from fpdf import FPDF

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base" / "guidelines"

DISCLAIMER = (
    "This is a synthetic, simplified reference document created for a portfolio "
    "software project. It is not real clinical guidance and must not be used for "
    "patient care."
)

GUIDELINES = {
    "diabetes_management": (
        "Type 2 Diabetes: Glucose and HbA1c Monitoring",
        [
            "Adults with type 2 diabetes should have fasting plasma glucose and "
            "hemoglobin A1c (HbA1c) checked at diagnosis and then at regular "
            "intervals thereafter. For patients meeting treatment goals with "
            "stable control, HbA1c testing every six months is generally "
            "sufficient. For patients with recent medication changes or "
            "inadequate control, testing every three months is recommended.",
            "A fasting glucose reference range of 70-99 mg/dL is considered "
            "normal. Values persistently above 126 mg/dL on fasting testing are "
            "consistent with diabetes. An HbA1c target below 7.0% is a "
            "reasonable general goal for most non-pregnant adults, though "
            "targets should be individualized based on age, comorbidities, "
            "hypoglycemia risk, and disease duration.",
            "Glucose values below 70 mg/dL indicate hypoglycemia and should "
            "prompt an immediate review of current medications, especially "
            "insulin and sulfonylureas. Values below 54 mg/dL represent "
            "clinically significant hypoglycemia requiring urgent attention. "
            "Severe hyperglycemia, generally above 400 mg/dL, or any glucose "
            "value accompanied by symptoms of diabetic ketoacidosis, warrants "
            "urgent clinical evaluation rather than routine follow-up.",
            "Renal function should be assessed periodically in patients with "
            "diabetes, since several glucose-lowering medications, including "
            "metformin, require dose adjustment or temporary discontinuation "
            "in the setting of impaired kidney function or iodinated contrast "
            "exposure.",
        ],
    ),
    "hypertension_management": (
        "Hypertension: Blood Pressure and Electrolyte Monitoring",
        [
            "Patients on antihypertensive therapy, particularly ACE inhibitors, "
            "angiotensin receptor blockers, and potassium-sparing diuretics such "
            "as spironolactone, should have serum potassium and creatinine "
            "checked within one to two weeks of starting or up-titrating "
            "therapy, and periodically thereafter.",
            "A serum potassium reference range of 3.5-5.1 mmol/L is considered "
            "normal. Values above 5.5 mmol/L represent hyperkalemia and should "
            "prompt medication review; values above 6.5 mmol/L are a critical "
            "result requiring urgent evaluation, including consideration of "
            "cardiac monitoring, given the risk of life-threatening arrhythmia.",
            "Combining an ACE inhibitor with a potassium supplement or a second "
            "potassium-sparing agent increases hyperkalemia risk and should be "
            "monitored closely rather than avoided outright when clinically "
            "indicated, since these combinations are sometimes necessary for "
            "adequate blood pressure or heart failure management.",
            "Sodium should also be monitored periodically, particularly in "
            "older adults or those on diuretics, as both hyponatremia and "
            "hypernatremia can present subtly with confusion, fatigue, or falls "
            "before becoming severe.",
        ],
    ),
    "anticoagulation_management": (
        "Anticoagulation: INR Monitoring and Drug Interactions",
        [
            "Patients on warfarin require regular INR monitoring, with "
            "frequency depending on stability of control: as often as twice "
            "weekly during initiation or dose changes, decreasing to every four "
            "weeks once a stable therapeutic range is achieved.",
            "A therapeutic INR range of 2.0-3.0 is typical for most "
            "indications, though certain conditions such as mechanical heart "
            "valves may require a higher target range. An INR reference range "
            "of 0.8-1.1 describes a patient not on anticoagulation; a critical "
            "result above 5.0 indicates a substantially elevated bleeding risk "
            "and warrants urgent clinical assessment.",
            "Several commonly prescribed medications meaningfully affect "
            "warfarin's anticoagulant effect. Aspirin combined with warfarin "
            "increases bleeding risk through an additive rather than "
            "pharmacokinetic mechanism and should be avoided unless the "
            "combination is specifically indicated. Amiodarone inhibits "
            "warfarin metabolism and typically requires a dose reduction of "
            "roughly one-third to one-half when co-administered, with closer "
            "INR monitoring during the transition.",
            "Any new medication, herbal supplement, or significant dietary "
            "change (particularly involving vitamin K intake) in a patient on "
            "warfarin should prompt a follow-up INR check rather than waiting "
            "for the next scheduled interval.",
        ],
    ),
    "ckd_monitoring": (
        "Chronic Kidney Disease: Creatinine, eGFR, and Medication Safety",
        [
            "Estimated glomerular filtration rate (eGFR) and serum creatinine "
            "are the primary markers used to stage and monitor chronic kidney "
            "disease. A reference eGFR of 90-120 mL/min/1.73m2 is considered "
            "normal; sustained values below 60 for three months or more are "
            "diagnostic of chronic kidney disease, and values below 15 "
            "represent kidney failure requiring urgent nephrology involvement.",
            "Creatinine values should be interpreted alongside the patient's "
            "baseline, since a 'normal range' value can still represent acute "
            "kidney injury in a patient whose baseline was previously lower. A "
            "critical creatinine result above 4.0 mg/dL should prompt urgent "
            "evaluation regardless of the patient's chronic baseline.",
            "Metformin is renally cleared, and its use in patients with "
            "reduced eGFR requires dose adjustment or discontinuation below "
            "certain thresholds due to the risk of lactic acidosis. This risk "
            "is further increased by exposure to iodinated contrast media, "
            "which can transiently worsen renal function; metformin should be "
            "held before and for 48 hours after contrast administration in at "
            "risk patients.",
            "Patients with chronic kidney disease on ACE inhibitors or "
            "potassium-sparing diuretics require closer potassium and "
            "creatinine monitoring than patients with normal renal function, "
            "since impaired excretion increases the risk of both hyperkalemia "
            "and further renal decline.",
        ],
    ),
}


def build_pdf(title: str, paragraphs: list[str]) -> FPDF:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(2)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, DISCLAIMER)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    for paragraph in paragraphs:
        pdf.multi_cell(0, 6, paragraph)
        pdf.ln(3)

    return pdf


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (title, paragraphs) in GUIDELINES.items():
        pdf = build_pdf(title, paragraphs)
        output_path = OUTPUT_DIR / f"{filename}.pdf"
        pdf.output(str(output_path))
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
