import { useState, useEffect, useCallback } from 'react';
import type { WebSocketMessage, TableData } from '../types/data';

interface UseWebSocketReturn {
    data: TableData[];
    isConnected: boolean;
    error: string | null;
    reconnect: () => void;
}

const useWebSocket = (url: string): UseWebSocketReturn => {
    const [data, setData] = useState<TableData[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [ws, setWs] = useState<WebSocket | null>(null);

    const connect = useCallback(() => {
        try {
            const websocket = new WebSocket(url);

            websocket.onopen = () => {
                console.log('WebSocket connected');
                setIsConnected(true);
                setError(null);

                // Request initial data
                websocket.send('get data');
            };

            websocket.onmessage = (event) => {
                try {
                    // Check if the message looks like JSON
                    const data = event.data.trim();
                    if (data.startsWith('{') || data.startsWith('[')) {
                        const message: WebSocketMessage = JSON.parse(data);
                        console.log('message', message);
                        // Transform the server response into TableData format programmatically
                        const tableData: TableData[] = [];

                        // Handle all timeframe data keys programmatically
                        Object.keys(message).forEach(key => {
                            // Skip non-dataframe keys
                            if (key === 'timestamp') {
                                return;
                            }

                            // Check if it's a timeframe data key (e.g., "1m_data", "3m_data", "5m_data", etc.)
                            if (key.endsWith('_data')) {
                                const timeframe = key.replace('_data', ''); // Extract "1m", "3m", "5m", etc.
                                const dataValue = message[key];

                                // Type guard to ensure it's DataFrameData
                                if (dataValue && typeof dataValue === 'object' && 'rows' in dataValue) {
                                    tableData.push({
                                        timeframe: timeframe,
                                        data: dataValue.rows,
                                        columns: dataValue.rows.length > 0
                                            ? Object.keys(dataValue.rows[0])
                                            : []
                                    });
                                }
                            }
                        });

                        setData(tableData);
                    } else {
                        console.log('Received non-JSON message:', event.data);
                    }
                } catch (parseError) {
                    console.error('Error parsing WebSocket message:', parseError);
                    console.log('Raw message that failed to parse:', event.data);
                    setError('Failed to parse incoming data');
                }
            };

            websocket.onclose = () => {
                console.log('WebSocket disconnected');
                setIsConnected(false);
            };

            websocket.onerror = (event) => {
                console.error('WebSocket error:', event);
                setError('WebSocket connection error');
                setIsConnected(false);
            };

            setWs(websocket);
        } catch (err) {
            console.error('Failed to create WebSocket connection:', err);
            setError('Failed to establish WebSocket connection');
        }
    }, [url]);

    const reconnect = useCallback(() => {
        if (ws) {
            ws.close();
        }
        connect();
    }, [ws, connect]);

    useEffect(() => {
        connect();

        // Set up periodic data requests (every 60 seconds)
        const intervalId = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                console.log('Requesting data...');
                ws.send('get data');
            }
        }, 1000);

        return () => {
            if (ws) {
                ws.close();
            }
            clearInterval(intervalId);
        };
    }, [connect]);

    return {
        data,
        isConnected,
        error,
        reconnect
    };
};

export default useWebSocket; 