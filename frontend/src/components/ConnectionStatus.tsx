import React from 'react';

interface ConnectionStatusProps {
    isConnected: boolean;
    error: string | null;
    onReconnect: () => void;
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
    isConnected,
    error,
    onReconnect
}) => {
    return (
        <div className="mb-6 p-4 rounded-lg border">
            <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                    <div
                        className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'
                            }`}
                    />
                    <span className="font-medium">
                        {isConnected ? 'Connected' : 'Disconnected'}
                    </span>
                </div>
                {!isConnected && (
                    <button
                        onClick={onReconnect}
                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                    >
                        Reconnect
                    </button>
                )}
            </div>
            {error && (
                <div className="mt-2 p-2 bg-red-100 border border-red-300 rounded text-red-700">
                    Error: {error}
                </div>
            )}
        </div>
    );
};

export default ConnectionStatus; 