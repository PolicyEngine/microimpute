import Papa from 'papaparse';
import { ImputationDataPoint, ImputationMetrics } from '@/types/imputation';

export function parseImputationCSV(csvContent: string): ImputationDataPoint[] {
  const result = Papa.parse<ImputationDataPoint>(csvContent, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  if (result.errors.length > 0) {
    console.warn('CSV parsing warnings:', result.errors);
    // Continue parsing despite warnings
  }

  // Return all parsed data - structure will depend on actual microimpute output
  return result.data.filter(row => row && Object.keys(row).length > 0);
}

export function getImputationMetrics(data: ImputationDataPoint[]): ImputationMetrics {
  if (data.length === 0) {
    return {
      totalRecords: 0,
      imputedCount: 0,
      variables: [],
      methods: [],
    };
  }

  // Extract unique variables if present
  const variables = data[0].variable
    ? [...new Set(data.map(d => d.variable).filter(Boolean) as string[])]
    : Object.keys(data[0]).filter(key => key !== 'id');

  // Extract unique methods if present
  const methods = data[0].method
    ? [...new Set(data.map(d => d.method).filter(Boolean) as string[])]
    : [];

  // Count imputed records (this logic will be updated based on actual data structure)
  const imputedCount = data.filter(d => d.imputed_value !== undefined).length;

  return {
    totalRecords: data.length,
    imputedCount,
    variables,
    methods,
  };
}