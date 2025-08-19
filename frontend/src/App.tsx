import React from 'react';
import './App.css';
import './index.css';
import useWebSocket from './hooks/useWebSocket';
import DataTable from './components/DataTable';
import ConnectionStatus from './components/ConnectionStatus';
import { config } from './config';

function App() {
  // Connect to the FastAPI backend websocket endpoint
  const { data, isConnected, error, reconnect } = useWebSocket(config.websocketUrl);

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Trading Data Dashboard
          </h1>
          <p className="text-lg text-gray-600">
            Real-time market data from multiple timeframes
          </p>
        </div>

        <ConnectionStatus
          isConnected={isConnected}
          error={error}
          onReconnect={reconnect}
        />

        {data.length === 0 && isConnected && (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg">
              Waiting for data...
            </div>
          </div>
        )}

        {data.length > 0 && (
          <div className="space-y-8">
            {data.map((tableData, index) => (
              <DataTable key={`${tableData.timeframe}-${index}`} tableData={tableData} />
            ))}
          </div>
        )}

        {!isConnected && !error && (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg">
              Connecting to server...
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
