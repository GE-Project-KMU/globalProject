export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public code: string = "REQUEST_FAILED",
  ) {
    super(message);
  }
}
export async function request<T>(
  token: string,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData))
    headers.set("Content-Type", "application/json");
  let response: Response;
  try {
    response = await fetch(path.startsWith("/api/") ? path : `/api/v1${path}`, {
      ...init,
      headers,
      cache: "no-store",
    });
  } catch {
    throw new ApiError(
      0,
      "서버에 연결하지 못했습니다. 서비스 실행 상태를 확인해 주세요.",
      "NETWORK_ERROR",
    );
  }
  if (response.status === 204) return undefined as T;
  if (!response.ok) {
    let payload: any = {};
    try {
      payload = await response.json();
    } catch {}
    const e = payload.error || {};
    throw new ApiError(
      response.status,
      e.message ||
        (typeof payload.detail === "string"
          ? payload.detail
          : `요청을 처리하지 못했습니다 (${response.status}).`),
      e.code,
    );
  }
  const payload = await response.json();
  return (payload.data ?? payload) as T;
}
export async function download(token: string, path: string, name: string) {
  const res = await fetch(path.startsWith("/api/") ? path : `/api/v1${path}`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new ApiError(
      res.status,
      "파일을 다운로드하지 못했습니다. 권한과 보관기한을 확인해 주세요.",
    );
  }
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
