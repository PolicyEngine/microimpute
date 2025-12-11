// Consistent color mapping for imputation methods across all charts
// Using Plotly color palette for consistency with Python visualizations

// Primary colors to use in order of appearance for the first 5 models
export const PRIMARY_COLORS = [
  '#636EFA', // Plotly blue
  '#EF553B', // Plotly red
  '#00CC96', // Plotly teal
  '#AB63FA', // Plotly purple
  '#FFA15A', // Plotly orange
];

// Additional fallback colors for when there are more than 5 models
export const FALLBACK_COLORS = [
  '#19D3F3', // Plotly cyan
  '#FF6692', // Plotly pink
  '#B6E880', // Plotly lime
  '#FF97FF', // Plotly magenta
  '#FECB52', // Plotly yellow
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
