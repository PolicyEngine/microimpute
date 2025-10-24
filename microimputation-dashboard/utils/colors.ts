// Consistent color mapping for imputation methods across all charts
// Using Plotly color palette for consistency with Python visualizations

export const METHOD_COLORS: Record<string, string> = {
  QRF: '#636EFA',           // Plotly blue
  OLS: '#EF553B',           // Plotly red
  QuantReg: '#00CC96',      // Plotly teal
  Matching: '#AB63FA',      // Plotly purple
  // Add more methods as needed
};

export const FALLBACK_COLORS = [
  '#FFA15A', // Plotly orange
  '#19D3F3', // Plotly cyan
  '#FF6692', // Plotly pink
  '#B6E880', // Plotly lime
  '#FF97FF', // Plotly magenta
  '#FECB52', // Plotly yellow
];

/**
 * Get color for a method, using predefined colors or fallback palette
 */
export function getMethodColor(method: string, index: number = 0): string {
  if (method in METHOD_COLORS) {
    return METHOD_COLORS[method];
  }
  return FALLBACK_COLORS[index % FALLBACK_COLORS.length];
}
