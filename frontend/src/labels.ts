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
