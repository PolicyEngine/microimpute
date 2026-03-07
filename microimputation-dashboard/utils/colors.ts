// Consistent color mapping for imputation methods across all charts
// Using design-system chart series colors as primary, Plotly palette as fallback
import { chartColors } from '@policyengine/design-system/charts';
import { colors } from '@policyengine/design-system/tokens';

// Primary colors from PE design-system chart series
export const PRIMARY_COLORS = [
  ...chartColors.series, // 5 PE chart series colors
];

// Additional fallback colors for when there are more than 5 models
export const FALLBACK_COLORS = [
  colors.blue[300],   // --pe-color-blue-300
  colors.error,       // --pe-color-error
  colors.success,     // --pe-color-success
  colors.warning,     // --pe-color-warning
  colors.primary[200], // --pe-color-primary-200
];

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
