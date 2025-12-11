# Connection Stability Improvements for TT02 Car

## Problem

The car connection was unstable, causing delays in command execution and potential loss of control.

## Solutions Implemented

### 1. **Safety Watchdog Timer**

- Auto-stops the car if no commands received for 2 seconds
- Prevents runaway situations due to connection loss
- Runs in background thread continuously monitoring last command time

### 2. **Command Retry Logic**

- Server automatically retries failed commands up to 2 times
- 50ms delay between retries
- Reduces impact of transient network errors

### 3. **Threaded HTTP Server**

- Uses `ThreadingMixIn` to handle multiple concurrent connections
- Each request processed in separate thread
- Request queue buffering for incoming connections during high load

### 4. **Socket Timeouts**

- 5-second timeout on socket operations
- Prevents hanging connections from blocking the server
- Graceful timeout handling with logging

### 5. **Keep-Alive Connections**

- HTTP keep-alive enabled for connection reuse
- Reduces overhead of establishing new connections
- Better performance over unstable networks

### 6. **Client-Side Retry Logic (Web UI)**

- JavaScript automatically retries failed commands (2 attempts)
- 100ms delay between client retries
- Commands repeated every 500ms while button held (compensates for packet loss)

### 7. **Connection Health Indicator**

- Visual indicator in web UI shows connection status:
  - **Green**: Good connection (< 2s since last success, < 3 failures)
  - **Orange**: Unstable connection (< 5s since last success, < 5 failures)
  - **Red**: Poor connection (> 5s or > 5 failures)

### 8. **Connection Statistics & Monitoring**

- Tracks total commands, failed commands, last client IP
- Auto-prints status every 10 seconds if there's activity
- Available at `/stats` endpoint for detailed view
- Shows stats on exit

### 9. **Error Handling & Logging**

- Comprehensive exception handling for network errors
- Logs timeouts, broken pipes, and connection resets
- Non-blocking error handling prevents crashes

### 10. **Graceful Shutdown**

- Proper cleanup of watchdog thread
- Statistics printed on exit
- Safe stop of all car operations

## Usage

### Start the Server

```bash
python tt02_app.py
```

### Monitor Connection Health

- Web UI shows real-time connection status with colored dot
- Terminal shows periodic stats (every 10 seconds if active)
- Visit `http://<car-ip>:6666/stats` for detailed statistics
- Visit `http://<car-ip>:6666/health` for quick health check

### Tips for Unstable Connections

1. **Hold buttons longer** - Commands repeat every 500ms while held
2. **Watch the health indicator** - Orange/red means connection issues
3. **Stay closer to the car** - Reduces Wi-Fi latency and packet loss
4. **Check terminal logs** - Shows timeout and retry messages
5. **Use `/stats` endpoint** - Monitor failure rate to diagnose issues

## Configuration

Adjust these values in `tt02_app.py` if needed:

```python
# Watchdog timeout (line ~95)
self.command_timeout = timedelta(seconds=2)  # Increase for slower networks

# Server timeouts (line ~285)
self.timeout = 5  # Socket timeout in seconds

# Retry settings (line ~345)
max_retries = 2  # Number of retry attempts
retry_delay = 0.05  # Delay between retries

# Client-side (HTML JavaScript, line ~235)
retries = 2  # Client retry attempts
timeout: 3000  # 3-second timeout
setInterval(() => send(pressCmd), 500)  # Command repeat interval
```

## Benefits

- ✅ Car automatically stops if connection lost (safety)
- ✅ Commands retry automatically on both server and client
- ✅ Visual feedback of connection quality
- ✅ Better handling of concurrent requests
- ✅ Reduced hanging/timeout issues
- ✅ Statistics for debugging connection problems
- ✅ More responsive over unstable Wi-Fi
