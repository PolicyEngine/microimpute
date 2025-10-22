// Type definitions for microimputation data
export interface ImputationDataPoint {
  type: string; // e.g., "benchmark_loss", "distribution_distance", "predictor_correlation"
  method: string; // e.g., "QRF", "OLS", "QuantReg", "Matching"
  variable: string; // e.g., "quantile_loss_mean_all", "log_loss_mean_all", or actual variable names
  quantile: string | number; // numeric (0.05, 0.1, etc.), "mean", or "N/A"
  metric_name: string; // e.g., "quantile_loss", "log_loss"
  metric_value: number | null; // numeric value of the metric
  split: string; // e.g., "train", "test", "full"
  additional_info: string; // JSON-formatted string with metadata
  [key: string]: any; // Allow additional fields
}

export interface ImputationMetrics {
  totalRecords: number;
  imputedCount: number;
  variables: string[];
  methods: string[];
}

export interface FileInfo {
  filename: string;
  loaded: boolean;
  data: ImputationDataPoint[];
}