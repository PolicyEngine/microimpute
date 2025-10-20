// Type definitions for microimputation data
export interface ImputationDataPoint {
  // Add fields based on what microimpute outputs
  // These are placeholder fields that will be updated based on actual CSV structure
  id?: string;
  variable?: string;
  original_value?: number;
  imputed_value?: number;
  method?: string;
  confidence?: number;
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