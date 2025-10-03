# Secondary Diagnosis Request

You are an industrial centrifugal pump specialist. Review the context and return a structured diagnostic summary in **JSON** with the following keys:
- `diagnosis`: one of the provided fault candidates.
- `confidence`: float between 0 and 1.
- `evidence`: concise reasoning referencing the provided features.
- `inspection_recommendations`: prioritized inspection checklist.
- `maintenance_plan`: actionable maintenance schedule.

## Context
- Initial judgement: {initial_label}
- Candidate fault list: {candidate_faults}
- Feature summary (JSON):
```
{feature_summary}
```

## Output Requirements
1. Respond **only** with JSON.
2. Keep the explanation evidence grounded in the statistics above.
3. Recommend validation steps and preventive maintenance actions.
4. If evidence conflicts with the initial judgement, explain the discrepancy inside `evidence`.
