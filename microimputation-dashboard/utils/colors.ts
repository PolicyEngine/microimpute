// Consistent color mapping for imputation methods across all charts
// Using colorblind-friendly Safe palette for consistency with Python visualizations

// Primary colors to use in order of appearance for the first 5 models
export const PRIMARY_COLORS = [
  '#88CCEE', // Cyan
  '#CC6677', // Rose
  '#DDCC77', // Sand
  '#117733', // Green
  '#332288', // Indigo
];

// Additional fallback colors for when there are more than 5 models
export const FALLBACK_COLORS = [
  '#AA4499', // Purple
  '#44AA99', // Teal
  '#999933', // Olive
  '#882255', // Wine
  '#661100', // Brown
];

// Chart styling constants matching Python backend PLOT_CONFIG
export const CHART_BG = '#FAFAFA';
export const GRID_COLOR = '#E5E5E5';
export const LINE_COLOR = '#CCCCCC';

/**
 * Get color for a method based on its index in the order of appearance.
 * First 5 models use PRIMARY_COLORS, additional models use FALLBACK_COLORS.
 */
export function getMethodColor(_method: string, index: number = 0): string {
  if (index < PRIMARY_COLORS.length) {
    return PRIMARY_COLORS[index];
  }
  const fallbackIndex = index - PRIMARY_COLORS.length;
  return FALLBACK_COLORS[fallbackIndex % FALLBACK_COLORS.length];
}
