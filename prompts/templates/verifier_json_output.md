Reply with exactly one JSON object and no prose or code fence.

Required JSON fields:
- verdict: one of the scenario decision labels
- confidence: number in [0.0, 1.0]
- event_class: event class being verified
- scenario_id: matched scenario id
- policy_id: policy pack id
- policy_version: policy pack version
- severity: severity derived from the scenario severity mapping
- evidence: array of concise observations supporting the verdict
- false_positive_reason: string or null
- recommended_action: action derived from the scenario action mapping

Use null for false_positive_reason unless the verdict is false_positive. Evidence must cite visible facts such as source location, persistence, movement, color, density, posture, occlusion, or frame quality when those fields are relevant to the scenario.
