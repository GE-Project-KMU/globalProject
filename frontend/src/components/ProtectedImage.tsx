import { useEffect, useState } from "react";
import { Eye } from "lucide-react";

export default function ProtectedImage({
  token,
  path,
}: {
  token: string;
  path?: string;
}) {
  const [url, setUrl] = useState(""),
    [error, setError] = useState(false);
  useEffect(() => {
    setUrl("");
    setError(false);
    if (!path) return;
    let alive = true,
      objectUrl = "";
    fetch(path.startsWith("/api/") ? path : `/api/v1${path}`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    })
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.blob();
      })
      .then((b) => {
        objectUrl = URL.createObjectURL(b);
        if (alive) setUrl(objectUrl);
      })
      .catch(() => {
        if (alive) setError(true);
      });
    return () => {
      alive = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [token, path]);
  return (
    <div className="evidence-image">
      {url ? (
        <img src={url} alt="비식별화된 노면 관측 사진" />
      ) : (
        <div>
          <Eye size={28} />
          <p>
            {error
              ? "사진을 불러오지 못했습니다"
              : path
                ? "사진을 불러오는 중입니다"
                : "등록된 근거 사진이 없습니다"}
          </p>
        </div>
      )}
    </div>
  );
}
