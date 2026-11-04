import { useEffect, useState } from "react";
import { ImagePlus, ShieldCheck } from "lucide-react";
import { ErrorBox } from "../components/UI";
import { parseReferencePack, type ReferencePack } from "../referencePack";

export default function ReferencePreview() {
  const [pack, setPack] = useState<ReferencePack | null>(null),
    [error, setError] = useState(""),
    [checked, setChecked] = useState(false),
    [opened, setOpened] = useState(false);
  useEffect(() => {
    if (!pack) return;
    const timer = window.setInterval(() => {
      if (Date.parse(pack.expires_at) <= Date.now()) {
        setPack(null);
        setOpened(false);
        setChecked(false);
        setError("보관 기한이 지나 참고사진 미리보기를 닫았습니다.");
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [pack]);
  async function select(file?: File) {
    setPack(null);
    setError("");
    setChecked(false);
    setOpened(false);
    if (!file) return;
    try {
      if (file.size > 5_000_000)
        throw new Error("참고사진 묶음은 5MB 이하여야 합니다.");
      setPack(parseReferencePack(await file.text(), file.size));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  function open() {
    if (!pack || !checked) return;
    try {
      parseReferencePack(
        JSON.stringify(pack),
        new Blob([JSON.stringify(pack)]).size,
      );
      setOpened(true);
    } catch (e) {
      setPack(null);
      setError((e as Error).message);
    }
  }
  return (
    <div className="narrow-section">
      <section className="panel management-form">
        <h2>실노면 로컬 참고사진</h2>
        <p>
          실제 도로 사진을 별도로 살펴보는 공간입니다. 관측 목록·지도·보고서에
          추가하지 않습니다.
        </p>
        <div className="notice">
          <ShieldCheck size={18} /> 개인정보 검수 미완료 · AI 분석 미실행 · 기관
          공유 승인 없음. 도로 픽셀을 기준으로 자른 사진도 사용자 검수와 공유
          승인을 대신하지 않습니다.
        </div>
        <p>
          프로젝트 폴더에서{" "}
          <strong>data → demo-reference → road-reference-pack.json</strong>을
          선택하세요. 사진은 이 브라우저 안에서만 열리며 서버로 업로드하지
          않습니다.
        </p>
        <label className="file-picker">
          <ImagePlus size={20} /> 로컬 참고사진 묶음 선택
          <input
            type="file"
            accept="application/json,.json"
            aria-label="로컬 참고사진 묶음"
            onChange={(e) => void select(e.target.files?.[0])}
          />
        </label>
        {error && <ErrorBox message={error} />}
        {pack && (
          <>
            <p>
              <b>출처:</b> {pack.source_name}
              <br />
              {pack.attribution}
              <br />
              <a
                href="https://creativecommons.org/licenses/by/4.0/"
                target="_blank"
                rel="noreferrer"
              >
                CC BY 4.0
              </a>{" "}
              · {pack.modifications}
            </p>
            <p>
              보관 기한:{" "}
              {new Intl.DateTimeFormat("ko-KR", {
                timeZone: "Asia/Seoul",
                year: "numeric",
                month: "2-digit",
                day: "2-digit",
                hour: "2-digit",
                minute: "2-digit",
              }).format(new Date(pack.expires_at))}{" "}
              (한국 표준시, UTC+09:00)
              <br />
              <small>
                기한 검사는 이 PC 시각을 기준으로 합니다. 원본 기한을 연장하지
                않습니다.
              </small>
            </p>
            {!opened && (
              <>
                <label className="checkbox-field">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => setChecked(e.target.checked)}
                  />{" "}
                  출처·보관 기한을 확인했으며 로컬 검수 목적으로만 열겠습니다.
                </label>
                <button
                  className="button primary"
                  disabled={!checked}
                  onClick={open}
                >
                  참고사진 로컬에서 열기
                </button>
              </>
            )}
            {opened && (
              <div className="reference-grid">
                {pack.images.map((photo, i) => (
                  <figure key={i}>
                    <img
                      src={photo.data_url}
                      alt={photo.title}
                      onLoad={(e) => {
                        if (
                          e.currentTarget.naturalWidth !== photo.width ||
                          e.currentTarget.naturalHeight !== photo.height
                        ) {
                          setOpened(false);
                          setError(
                            "사진 크기가 파일 정보와 일치하지 않습니다.",
                          );
                        }
                      }}
                      onError={() => {
                        setOpened(false);
                        setError("사진을 읽지 못했습니다.");
                      }}
                    />
                    <figcaption>
                      <b>{photo.title}</b>
                      <small>
                        실노면 참고사진 · 검수 미완료 · AI 검출 아님
                      </small>
                    </figcaption>
                  </figure>
                ))}
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
