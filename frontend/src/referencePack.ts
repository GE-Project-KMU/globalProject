export interface ReferencePack {
  schema_version: 1;
  pack_name: string;
  source_name: string;
  attribution: string;
  license_id: "CC-BY-4.0";
  modifications: string;
  expires_at: string;
  source_expires_at: string;
  local_only: true;
  human_privacy_reviewed: false;
  inference_performed: false;
  images: { title: string; data_url: string; width: number; height: number }[];
}
const sourceExpiry = Date.parse("2027-02-01T16:51:19.893991+00:00");
function jpegSize(raw: string): [number, number] {
  const byte = (i: number) => raw.charCodeAt(i),
    word = (i: number) => byte(i) * 256 + byte(i + 1);
  let i = 2;
  while (i + 4 < raw.length) {
    if (byte(i++) !== 255) break;
    while (byte(i) === 255) i++;
    const marker = byte(i++);
    if (marker === 0xda || marker === 0xd9) break;
    if (marker === 1 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    const length = word(i);
    if (length < 2 || i + length > raw.length) break;
    if (
      [
        0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce,
        0xcf,
      ].includes(marker)
    ) {
      if (length < 8) break;
      return [word(i + 5), word(i + 3)];
    }
    i += length;
  }
  throw new Error("JPEG 사진의 크기를 확인할 수 없습니다.");
}
function exactKeys(
  value: unknown,
  keys: string[],
): asserts value is Record<string, any> {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    Object.keys(value).sort().join("|") !== [...keys].sort().join("|")
  )
    throw new Error("지원하지 않는 참고사진 파일 형식입니다.");
}
export function parseReferencePack(
  text: string,
  bytes: number,
  now = Date.now(),
): ReferencePack {
  if (bytes > 5_000_000 || new TextEncoder().encode(text).length > 5_000_000)
    throw new Error("참고사진 묶음은 5MB 이하여야 합니다.");
  let pack: any;
  try {
    pack = JSON.parse(text);
  } catch {
    throw new Error("유효한 JSON 참고사진 묶음을 선택하세요.");
  }
  exactKeys(pack, [
    "schema_version",
    "pack_name",
    "source_name",
    "attribution",
    "license_id",
    "modifications",
    "expires_at",
    "source_expires_at",
    "local_only",
    "human_privacy_reviewed",
    "inference_performed",
    "images",
  ]);
  if (
    pack.schema_version !== 1 ||
    pack.license_id !== "CC-BY-4.0" ||
    pack.local_only !== true ||
    pack.human_privacy_reviewed !== false ||
    pack.inference_performed !== false
  )
    throw new Error(
      "로컬 연구 참고사진 형식만 열 수 있습니다. 공유 승인이나 AI 검출 결과를 가져올 수 없습니다.",
    );
  if (
    pack.source_name !==
      "Road Traversing Knowledge (RTK), UFSC / Mendeley Data" ||
    pack.attribution !==
      "Thiago Rateke, Aldo von Wangenheim, Rafael Toledo · DOI 10.17632/hssswvmjwf.1"
  )
    throw new Error(
      "출처와 저작자 정보가 일치하는 RTK 참고사진 묶음이 필요합니다.",
    );
  for (const key of ["pack_name", "modifications"])
    if (
      typeof pack[key] !== "string" ||
      !pack[key].trim() ||
      pack[key].length > 500 ||
      /https?:|file:|[/\\]/i.test(pack[key])
    )
      throw new Error(
        "사진 경로나 외부 주소가 포함된 참고사진 파일은 열 수 없습니다.",
      );
  for (const key of ["expires_at", "source_expires_at"])
    if (
      typeof pack[key] !== "string" ||
      !/(Z|[+-]\d{2}:\d{2})$/.test(pack[key]) ||
      !Number.isFinite(Date.parse(pack[key]))
    )
      throw new Error("시간대가 명시된 보관 기한이 필요합니다.");
  if (
    Date.parse(pack.expires_at) <= now ||
    Date.parse(pack.expires_at) > Date.parse(pack.source_expires_at) ||
    Date.parse(pack.source_expires_at) > sourceExpiry
  )
    throw new Error("보관 기한이 지났거나 원본 보관 기한을 넘긴 파일입니다.");
  if (
    !Array.isArray(pack.images) ||
    !pack.images.length ||
    pack.images.length > 6
  )
    throw new Error("참고사진은 1~6장만 열 수 있습니다.");
  for (const photo of pack.images) {
    exactKeys(photo, ["title", "data_url", "width", "height"]);
    if (
      typeof photo.title !== "string" ||
      !photo.title.trim() ||
      photo.title.length > 100 ||
      /[/\\]|https?:|file:/i.test(photo.title)
    )
      throw new Error("유효한 사진 제목이 필요합니다.");
    if (
      ![photo.width, photo.height].every(
        (x) => Number.isInteger(x) && x > 0 && x <= 4096,
      )
    )
      throw new Error("사진 크기는 각 4,096픽셀 이하여야 합니다.");
    if (
      typeof photo.data_url !== "string" ||
      photo.data_url.length > 1_000_000 ||
      !/^data:image\/jpeg;base64,[A-Za-z0-9+/]+={0,2}$/.test(photo.data_url)
    )
      throw new Error(
        "사진은 파일 안에 포함된 JPEG만 허용합니다. 외부 URL이나 경로는 사용할 수 없습니다.",
      );
    let raw = "";
    try {
      raw = atob(photo.data_url.split(",")[1]);
    } catch {
      throw new Error("JPEG 사진을 읽지 못했습니다.");
    }
    if (
      raw.length < 4 ||
      raw.charCodeAt(0) !== 255 ||
      raw.charCodeAt(1) !== 216 ||
      raw.charCodeAt(raw.length - 2) !== 255 ||
      raw.charCodeAt(raw.length - 1) !== 217
    )
      throw new Error("유효한 JPEG 사진이 아닙니다.");
    const [width, height] = jpegSize(raw);
    if (
      width !== photo.width ||
      height !== photo.height ||
      width * height > 1_000_000
    )
      throw new Error(
        "사진의 실제 크기가 일치하지 않거나 허용 크기를 초과했습니다.",
      );
  }
  return pack as ReferencePack;
}
