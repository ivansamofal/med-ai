from app.eval.experiment import ExperimentVariant, run_experiment
from app.eval.models import GoldenQuestion
from app.knowledge.query_engine import RetrievedPassage

QUESTION = GoldenQuestion(
    id="q1",
    question="What is the glucose reference range?",
    expected_source_titles=["Glucose reference range"],
    required_keywords=["70", "99"],
)

FULL_PASSAGE = RetrievedPassage(
    text="Glucose reference range is 70-99 mg/dL.",
    source_title="Glucose reference range",
    source_type="reference_range",
    score=0.9,
)
UNRELATED_PASSAGE = RetrievedPassage(
    text="Warfarin and Aspirin together increase bleeding risk.",
    source_title="Warfarin / Aspirin interaction",
    source_type="drug_interaction",
    score=0.5,
)


def test_run_experiment_reports_a_real_pass_rate_difference_between_variants():
    variants = [
        ExperimentVariant(name="misses", retrieve_fn=lambda q: [UNRELATED_PASSAGE]),
        ExperimentVariant(name="finds_it", retrieve_fn=lambda q: [UNRELATED_PASSAGE, FULL_PASSAGE]),
    ]

    reports = run_experiment(variants, [QUESTION])

    assert reports[0].pass_rate == 0.0
    assert reports[1].pass_rate == 1.0
    assert not reports[0].results[0].passed
    assert reports[1].results[0].passed


def test_run_experiment_preserves_variant_order():
    variants = [
        ExperimentVariant(name="a", retrieve_fn=lambda q: []),
        ExperimentVariant(name="b", retrieve_fn=lambda q: []),
    ]

    reports = run_experiment(variants, [QUESTION])

    assert [r.name for r in reports] == ["a", "b"]


def test_variant_report_pass_rate_handles_empty_results():
    from app.eval.experiment import VariantReport

    report = VariantReport(name="empty", results=[])

    assert report.pass_rate == 0.0
