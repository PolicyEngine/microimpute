'use client';

import { useMemo, useState } from 'react';
import { ImputationDataPoint } from '@/types/imputation';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface DistributionOverlayProps {
  data: ImputationDataPoint[];
}

interface BinData {
  binIndex: number;
  binStart: number;
  binEnd: number;
  donorHeight: number;
  receiverHeight: number;
  binLabel?: string;
}

interface CategoryData {
  category: string;
  donorProportion: number;
  receiverProportion: number;
}

interface VariableDistribution {
  variable: string;
  type: 'numerical' | 'categorical';
  data: BinData[] | CategoryData[];
  nSamplesDonor: number;
  nSamplesReceiver: number;
}

export default function DistributionOverlay({
  data,
}: DistributionOverlayProps) {
  // Extract distribution bins data
  const distributionBins = useMemo(() => {
    return data.filter((d) => d.type === 'distribution_bins');
  }, [data]);

  // Parse and group distribution data by variable
  const variableDistributions = useMemo(() => {
    const distributions: Record<string, VariableDistribution> = {};

    distributionBins.forEach((d) => {
      const variable = d.variable;

      if (!distributions[variable]) {
        distributions[variable] = {
          variable,
          type:
            d.metric_name === 'histogram_distribution'
              ? 'numerical'
              : 'categorical',
          data: [],
          nSamplesDonor: 0,
          nSamplesReceiver: 0,
        };
      }

      try {
        const info = JSON.parse(d.additional_info);

        if (d.metric_name === 'histogram_distribution') {
          // Numerical variable
          (distributions[variable].data as BinData[]).push({
            binIndex: info.bin_index,
            binStart: info.bin_start,
            binEnd: info.bin_end,
            donorHeight: info.donor_height,
            receiverHeight: info.receiver_height,
            binLabel: `${info.bin_start.toFixed(2)}-${info.bin_end.toFixed(2)}`,
          });
          distributions[variable].nSamplesDonor = info.n_samples_donor;
          distributions[variable].nSamplesReceiver = info.n_samples_receiver;
        } else if (d.metric_name === 'categorical_distribution') {
          // Categorical variable
          (distributions[variable].data as CategoryData[]).push({
            category: info.category,
            donorProportion: info.donor_proportion,
            receiverProportion: info.receiver_proportion,
          });
          distributions[variable].nSamplesDonor = info.n_samples_donor;
          distributions[variable].nSamplesReceiver = info.n_samples_receiver;
        }
      } catch (e) {
        console.error('Error parsing distribution bin data:', e);
      }
    });

    // Sort numerical bins by bin index
    Object.values(distributions).forEach((dist) => {
      if (dist.type === 'numerical') {
        (dist.data as BinData[]).sort((a, b) => a.binIndex - b.binIndex);
      }
    });

    return distributions;
  }, [distributionBins]);

  const variables = Object.keys(variableDistributions);
  const [selectedVariable, setSelectedVariable] = useState<string>(
    variables[0] || ''
  );

  if (variables.length === 0) {
    return null;
  }

  const selectedDistribution = variableDistributions[selectedVariable];

  const renderNumericalDistribution = (dist: VariableDistribution) => {
    const chartData = (dist.data as BinData[]).map((bin) => ({
      name: bin.binLabel,
      Donor: bin.donorHeight,
      Receiver: bin.receiverHeight,
      binStart: bin.binStart,
      binEnd: bin.binEnd,
    }));

    return (
      <div>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="name"
              angle={-45}
              textAnchor="end"
              height={80}
              tick={{ fill: '#000000', fontSize: 11 }}
              label={{
                value: `${dist.variable} (binned values)`,
                position: 'insideBottom',
                offset: -50,
                style: { fill: '#000000' },
              }}
            />
            <YAxis
              tick={{ fill: '#000000' }}
              label={{
                value: 'Percentage (%)',
                angle: -90,
                position: 'insideLeft',
                offset: 10,
                style: { fill: '#000000', textAnchor: 'middle' },
              }}
            />
            <Tooltip
              formatter={(value: number) => [`${value.toFixed(2)}%`, '']}
              labelFormatter={(label) => `Bin: ${label}`}
              contentStyle={{ color: '#000000' }}
              labelStyle={{ color: '#000000' }}
            />
            <Legend wrapperStyle={{ color: '#000000', paddingTop: '10px' }} />
            <Bar
              dataKey="Donor"
              fill="#3b82f6"
              fillOpacity={0.7}
              name={`Donor (n=${dist.nSamplesDonor})`}
            />
            <Bar
              dataKey="Receiver"
              fill="#ef4444"
              fillOpacity={0.7}
              name={`Receiver (n=${dist.nSamplesReceiver})`}
            />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-600 mt-2 text-center">
          Histogram with {(dist.data as BinData[]).length} bins. Each bin shows the percentage of values falling within that range.
          Overlapping bars indicate similar distributions.
        </p>
      </div>
    );
  };

  const renderCategoricalDistribution = (dist: VariableDistribution) => {
    const chartData = (dist.data as CategoryData[]).map((cat) => ({
      category: cat.category,
      Donor: cat.donorProportion,
      Receiver: cat.receiverProportion,
    }));

    return (
      <div>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="category"
              tick={{ fill: '#000000', fontSize: 12 }}
              label={{
                value: `${dist.variable} (categories)`,
                position: 'insideBottom',
                offset: -10,
                style: { fill: '#000000' },
              }}
            />
            <YAxis
              tick={{ fill: '#000000' }}
              label={{
                value: 'Percentage (%)',
                angle: -90,
                position: 'insideLeft',
                offset: 10,
                style: { fill: '#000000', textAnchor: 'middle' },
              }}
            />
            <Tooltip
              formatter={(value: number) => [`${value.toFixed(2)}%`, '']}
              contentStyle={{ color: '#000000' }}
              labelStyle={{ color: '#000000' }}
            />
            <Legend wrapperStyle={{ color: '#000000', paddingTop: '10px' }} />
            <Bar
              dataKey="Donor"
              fill="#3b82f6"
              fillOpacity={0.7}
              name={`Donor (n=${dist.nSamplesDonor})`}
            />
            <Bar
              dataKey="Receiver"
              fill="#ef4444"
              fillOpacity={0.7}
              name={`Receiver (n=${dist.nSamplesReceiver})`}
            />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-600 mt-2 text-center">
          Side-by-side bars compare the proportion of each category in donor vs receiver data.
        </p>
      </div>
    );
  };

  return (
    <div className="mb-8 p-6 bg-gradient-to-br from-indigo-50 to-purple-50 border border-indigo-200 rounded-lg">
      <div className="mb-4">
        <h3 className="text-xl font-semibold mb-2 text-gray-900">
          Distribution comparison
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Visual comparison of donor and receiver distributions for imputed
          variables. Overlapping distributions indicate successful imputation.
        </p>

        {/* Variable selector */}
        {variables.length > 1 && (
          <div className="flex items-center gap-3">
            <label
              htmlFor="variable-select"
              className="text-sm font-medium text-gray-700"
            >
              Select variable:
            </label>
            <select
              id="variable-select"
              value={selectedVariable}
              onChange={(e) => setSelectedVariable(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white text-gray-900"
            >
              {variables.map((variable) => (
                <option key={variable} value={variable}>
                  {variable} (
                  {variableDistributions[variable].type === 'numerical'
                    ? 'numerical'
                    : 'categorical'}
                  )
                </option>
              ))}
            </select>
          </div>
        )}

        {variables.length === 1 && (
          <div className="text-sm text-gray-700">
            <span className="font-medium">Variable:</span>{' '}
            <span className="font-mono">{selectedVariable}</span>{' '}
            <span className="text-gray-500">
              ({selectedDistribution?.type})
            </span>
          </div>
        )}
      </div>

      {/* Render appropriate chart */}
      {selectedDistribution && (
        <div className="bg-white p-4 rounded-lg">
          {selectedDistribution.type === 'numerical'
            ? renderNumericalDistribution(selectedDistribution)
            : renderCategoricalDistribution(selectedDistribution)}
        </div>
      )}
    </div>
  );
}
