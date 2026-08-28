// Thin WebSocket client for the live plate stream, with basic auto-reconnect.
// STATUS: step-1 plumbing — talks to backend/ws/plate_stream.py's random-walk
// stub today; the message shape (PlateStateMessage) doesn't change when the
// real biology/sensor/GNN pipeline replaces the stub later.

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/plate";
const RECONNECT_DELAY_MS = 2000;

export function connectPlateStream({ onMessage, onStatusChange }) {
  let socket = null;
  let closedByCaller = false;

  function open() {
    socket = new WebSocket(WS_URL);

    socket.onopen = () => onStatusChange?.("connected");
    socket.onclose = () => {
      onStatusChange?.("disconnected");
      if (!closedByCaller) setTimeout(open, RECONNECT_DELAY_MS);
    };
    socket.onerror = () => socket.close();
    socket.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data));
      } catch (err) {
        console.error("Failed to parse plate stream message", err);
      }
    };
  }

  open();

  return () => {
    closedByCaller = true;
    socket?.close();
  };
}
