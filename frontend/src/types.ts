export interface Observation {
  id?: string;
  observation_id: string;
  event_id: string;
  case_id: string;
  case_version: number;
  type: "pothole" | "lane_segment";
  captured_at: string;
  location: {
    lat: number | null;
    lon: number | null;
    quality?: string;
    source?: string;
  };
  review: { status: string; priority: string; reason?: string };
  pothole?: {
    confidence: number;
    bbox_xywh_px?: number[];
    track_observations?: number;
    candidate_state?: string;
  } | null;
  lane?: {
    retention_pct: number | null;
    grade: string;
    valid_area_ratio: number | null;
    segment_support_ratio?: number | null;
    unassessable_reason?: string | null;
    color?: string;
    line_type?: string;
    paint_mask_rle?: { size: number[]; counts: number[] };
    outline_polygon_px?: number[][];
  } | null;
  quality?: { usable: boolean; reasons: string[] };
  image_content_path?: string;
  image?: { file_id?: string };
  expires_at?: string;
  model_version?: string;
  profile_id?: string;
  demo?: boolean;
  synthetic?: boolean;
  corrections?: unknown[];
}
export interface ListResult {
  items: Observation[];
  next_cursor?: string | null;
  counts_by_type?: Record<string, number>;
}
export interface User {
  id: string;
  login?: string;
  role: string;
  org_id?: string;
}
export interface Summary {
  trend_by_type?: Record<
    string,
    { date: string; candidates: number; confirmed: number; completed: number }[]
  >;
  trend?: {
    date: string;
    candidates: number;
    confirmed: number;
    completed: number;
  }[];
  trend_counts_unit?: string;
  total: number;
  pending: number;
  potholes: number;
  worn_lanes: number;
  completed: number;
  unassessable: number;
}
