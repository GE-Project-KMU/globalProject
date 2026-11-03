import type { Observation } from "./types";
export const statusLabels: Record<string, string> = {
  unreviewed: "미확인",
  confirmed: "확인됨",
  false_positive: "오탐",
  recapture: "재촬영",
  action_planned: "조치 예정",
  completed: "조치 완료",
};
export const gradeLabels: Record<string, string> = {
  visible_good: "시각상 양호",
  monitor: "관찰",
  worn_candidate: "마모 후보",
  unassessable: "판정 불가",
};
const privacyResults: Record<string, string> = {
  MANUAL_ADMIN_REVIEW_REQUIRED: "관리자 검토 대기",
  DELETION_VERIFIED: "삭제 검증 완료",
  ACCESS_APPROVED: "열람 승인 완료",
  CORRECTION_APPLIED: "정정 반영 완료",
  ADMIN_REJECTED: "관리자 반려",
  ACCESS_DATA_DELETED: "자료 삭제로 열람 만료",
  ACCESS_DATA_EXPIRED: "보관기간 만료로 열람 종료",
  ACCESS_EXPIRED: "열람 제공 기간 만료",
  DATA_EXPIRED: "원자료 보관기간 만료",
};
export const privacyResultLabel = (code?: string | null) =>
  code ? privacyResults[code] || "처리 내역 확인 필요" : "결과 대기";
const analysisReasons: Record<string, string> = {
  MODEL_NOT_READY: "분석에 필요한 학습 가중치가 준비되지 않았습니다.",
  WORKFLOW_FAILED:
    "분석을 완료하지 못했습니다. 준비 설정과 로컬 실행 기록을 확인하세요.",
  WORKFLOW_TIMEOUT:
    "분석 실행 시간이 초과되었습니다. 입력 크기와 장치 상태를 확인하세요.",
  SERVICE_RESTARTED:
    "서비스 재시작으로 분석이 중단되었습니다. 준비 상태를 확인하고 다시 요청하세요.",
};
export const analysisReasonLabel = (code?: string | null) =>
  code
    ? analysisReasons[code] || "분석 준비 상태와 운영 기록을 확인하세요."
    : "";
export const date = (value?: string) =>
  value
    ? new Intl.DateTimeFormat("ko-KR", {
        timeZone: "Asia/Seoul",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      }).format(new Date(value))
    : "—";
export const id = (o: Observation) => o.observation_id || o.id || "";
export const typeName = (o: Observation) =>
  o.type === "pothole" ? "포트홀" : "차선 마모";
