import React from 'react';
import type { TableData } from '../types/data.ts';
import { config } from '../config';

interface DataTableProps {
    tableData: TableData;
}

const DataTable: React.FC<DataTableProps> = ({ tableData }) => {
    const { timeframe, data, columns } = tableData;

    // Get the last N rows based on config
    const lastNRows = data.slice(-config.maxTableRows);

    return (
        <div className="mb-8">
            <h2 className="text-2xl font-bold mb-4 text-gray-800">
                Timeframe: {timeframe}
            </h2>
            <div className="overflow-x-auto shadow-lg rounded-lg">
                <table className="min-w-full bg-white border border-gray-300">
                    <thead className="bg-gray-50">
                        <tr>
                            {columns.map((column, index) => (
                                <th
                                    key={index}
                                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-200"
                                >
                                    {column}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {lastNRows.map((row, rowIndex) => (
                            <tr
                                key={rowIndex}
                                className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
                            >
                                {columns.map((column, colIndex) => (
                                    <td
                                        key={colIndex}
                                        className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 border-b border-gray-200"
                                    >
                                        {row[column] !== null && row[column] !== undefined
                                            ? String(row[column])
                                            : '-'}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="mt-2 text-sm text-gray-600">
                Showing last {lastNRows.length} rows of {data.length} total rows
            </div>
        </div>
    );
};

export default DataTable; 