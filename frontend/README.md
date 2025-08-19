# Trading Data Dashboard Frontend

A React-based frontend application that connects to a FastAPI backend via WebSocket to display real-time trading data in tables organized by timeframe.

## Features

- **Real-time WebSocket Connection**: Connects to FastAPI backend websocket endpoint
- **Multiple Timeframe Tables**: Displays data for each timeframe in separate tables
- **Last 15 Rows Display**: Shows the most recent 15 rows of data for each timeframe
- **Connection Status**: Visual indicator of websocket connection status
- **Error Handling**: Displays connection errors and provides reconnect functionality
- **Responsive Design**: Modern UI with Tailwind CSS styling

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- FastAPI backend running on `localhost:8000` with websocket endpoint at `/ws`

## Installation

1. Install dependencies:

```bash
npm install
```

2. Start the development server:

```bash
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

## Configuration

You can modify the configuration in `src/config.ts`:

- `websocketUrl`: WebSocket endpoint URL (default: `ws://localhost:8000/ws`)
- `maxTableRows`: Number of rows to display in each table (default: 15)
- `reconnectInterval`: Reconnection interval in milliseconds (default: 5000)
- `maxReconnectAttempts`: Maximum number of reconnection attempts (default: 5)

## Expected WebSocket Message Format

The frontend expects the following JSON message format from the backend:

```json
{
  "dataframes": [
    {
      "timeframe": "1m",
      "data": [
        {
          "timestamp": "2024-01-01T00:00:00",
          "open": 100.0,
          "high": 101.0,
          "low": 99.0,
          "close": 100.5,
          "volume": 1000
        }
      ],
      "columns": ["timestamp", "open", "high", "low", "close", "volume"]
    }
  ],
  "timestamp": "2024-01-01T00:00:00"
}
```

## Project Structure

```
src/
├── components/
│   ├── DataTable.tsx      # Table component for displaying dataframe data
│   └── ConnectionStatus.tsx # WebSocket connection status indicator
├── hooks/
│   └── useWebSocket.ts    # Custom hook for WebSocket management
├── types/
│   └── data.ts           # TypeScript type definitions
├── config.ts             # Application configuration
├── App.tsx               # Main application component
└── main.tsx              # Application entry point
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Technologies Used

- React 19
- TypeScript
- Tailwind CSS
- Vite
- WebSocket API
