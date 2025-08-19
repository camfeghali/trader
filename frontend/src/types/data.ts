export interface DataFrameRow {
    [key: string]: string | number | boolean | null;
}

export interface DataFrameData {
    count: number;
    rows: DataFrameRow[];
}

export interface WebSocketMessage {
    '1m_data': DataFrameData;
    '3m_data'?: DataFrameData;
    '5m_data'?: DataFrameData;
    '15m_data'?: DataFrameData;
    '30m_data'?: DataFrameData;
    '1h_data'?: DataFrameData;
    '4h_data'?: DataFrameData;
    '1d_data'?: DataFrameData;
    timestamp: string;
    [key: string]: DataFrameData | string | undefined; // Allow dynamic timeframe keys
}

export interface TableData {
    timeframe: string;
    data: DataFrameRow[];
    columns: string[];
} 