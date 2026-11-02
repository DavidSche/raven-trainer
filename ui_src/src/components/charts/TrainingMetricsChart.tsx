import React from 'react';
import { useTranslation } from 'react-i18next';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';

interface MetricsDataPoint {
  epoch: number;
  mAP50?: number;
  mAP50_95?: number;
  precision?: number;
  recall?: number;
  loss?: number;
}

export function TrainingMetricsChart({ data }: { data: MetricsDataPoint[] }) {
  const { t } = useTranslation();
  if (!data || data.length === 0) {
    return <div className="text-gray-400 text-sm py-12 text-center">{t('trainingDetail.noEpochData')}</div>;
  }

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          {data[0]?.mAP50 !== undefined && (
            <Line type="monotone" dataKey="mAP50" stroke="#4c6ef5" strokeWidth={2} dot={false} name="mAP50" />
          )}
          {data[0]?.mAP50_95 !== undefined && (
            <Line type="monotone" dataKey="mAP50_95" stroke="#7950f2" strokeWidth={2} dot={false} name="mAP50-95" />
          )}
          {data[0]?.precision !== undefined && (
            <Line type="monotone" dataKey="precision" stroke="#40c057" strokeWidth={2} dot={false} name="Precision" />
          )}
          {data[0]?.recall !== undefined && (
            <Line type="monotone" dataKey="recall" stroke="#fab005" strokeWidth={2} dot={false} name="Recall" />
          )}
          {data[0]?.loss !== undefined && (
            <Line type="monotone" dataKey="loss" stroke="#e64980" strokeWidth={2} dot={false} name="Loss" />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export type { MetricsDataPoint };
