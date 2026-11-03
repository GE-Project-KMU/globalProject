import { Route, TriangleAlert } from "lucide-react";
import { statusLabels, gradeLabels } from "../labels";
export function Badge({ status }: { status: string }) {
  return (
    <span className={`badge ${status}`}>
      {statusLabels[status] || gradeLabels[status] || status}
    </span>
  );
}

export function Empty({
  title = "아직 관측 자료가 없습니다",
  text = "장치에서 촬영·분석한 자료를 전송하면 이곳에서 검토할 수 있습니다.",
  action,
}: {
  title?: string;
  text?: string;
  action?: () => void;
}) {
  return (
    <div className="empty">
      <Route size={30} />
      <h3>{title}</h3>
      <p>{text}</p>
      {action && (
        <button className="button secondary" onClick={action}>
          필터 초기화
        </button>
      )}
    </div>
  );
}

export function ErrorBox({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="error-box" role="alert">
      <TriangleAlert size={18} />
      <span>{message}</span>
      {onRetry && <button onClick={onRetry}>다시 시도</button>}
    </div>
  );
}
