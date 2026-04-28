You are verifying a visual event candidate for a CCTV safety system.

Policy: {{ policy_id }} v{{ policy_version }}
Scenario: {{ scenario_id }}
Event class: {{ event_class }}
Detector labels: {{ detector_labels }}

Candidate metadata:
- detector label: {{ detector_label }}
- detector confidence: {{ detector_confidence }}
- stream: {{ stream_id }}
- site: {{ site_id }}
- camera: {{ camera_id }}
- zones: {{ zone_ids }}
- track id: {{ track_id }}
- time range: {{ start_pts_s }}s to {{ peak_pts_s }}s
- sampled keyframes: {{ keyframe_pts }}
- scenario context window: {{ context_window_s }}s

Normal or benign conditions:
{{ normal_conditions }}

Abnormal conditions that support a true alert:
{{ abnormal_conditions }}

False-positive hints:
{{ false_positive_hints }}

True-positive hints:
{{ true_positive_hints }}

Required evidence to cite:
{{ required_evidence }}

Return uncertain when:
{{ uncertain_when }}

Severity mapping:
{{ severity_mapping }}

Recommended action mapping:
{{ recommended_action_mapping }}

Decision labels: {{ verifier_output_labels }}

Use the frames in order. Decide whether this candidate matches the scenario-specific event semantics, not just whether the detector label appears plausible.

{{ json_output_requirements }}
