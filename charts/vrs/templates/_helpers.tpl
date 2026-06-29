{{- define "vrs.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "vrs.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "vrs.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "vrs.labels" -}}
app.kubernetes.io/name: {{ include "vrs.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "vrs.objectStorageEnv" -}}
- name: VRS_OBJECT_STORE
  value: {{ .Values.objectStorage.mode | quote }}
{{- if eq .Values.objectStorage.mode "local-pvc" }}
- name: VRS_OBJECT_STORE_ROOT
  value: {{ .Values.objectStorage.mountPath | quote }}
{{- else if eq .Values.objectStorage.mode "seaweedfs" }}
- name: VRS_OBJECT_STORE_ENDPOINT
  value: "http://{{ include "vrs.fullname" . }}-seaweedfs:{{ .Values.objectStorage.seaweedfs.s3Port }}"
- name: VRS_OBJECT_STORE_BUCKET
  value: {{ .Values.objectStorage.seaweedfs.bucket | quote }}
- name: VRS_OBJECT_STORE_REGION
  value: {{ .Values.objectStorage.seaweedfs.region | quote }}
- name: VRS_OBJECT_STORE_PREFIX
  value: {{ .Values.objectStorage.seaweedfs.prefix | quote }}
- name: VRS_OBJECT_STORE_FORCE_PATH_STYLE
  value: "true"
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ include "vrs.fullname" . }}-seaweedfs
      key: accessKey
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "vrs.fullname" . }}-seaweedfs
      key: secretKey
{{- else if eq .Values.objectStorage.mode "external" }}
- name: VRS_OBJECT_STORE_ENDPOINT
  value: {{ .Values.objectStorage.external.endpoint | quote }}
- name: VRS_OBJECT_STORE_BUCKET
  value: {{ .Values.objectStorage.external.bucket | quote }}
- name: VRS_OBJECT_STORE_REGION
  value: {{ .Values.objectStorage.external.region | quote }}
- name: VRS_OBJECT_STORE_PREFIX
  value: {{ .Values.objectStorage.external.prefix | quote }}
- name: VRS_OBJECT_STORE_FORCE_PATH_STYLE
  value: {{ .Values.objectStorage.external.forcePathStyle | quote }}
{{- if .Values.objectStorage.external.existingSecret }}
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ .Values.objectStorage.external.existingSecret | quote }}
      key: {{ .Values.objectStorage.external.accessKeySecretKey | quote }}
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.objectStorage.external.existingSecret | quote }}
      key: {{ .Values.objectStorage.external.secretKeySecretKey | quote }}
{{- end }}
{{- end }}
{{- end -}}
