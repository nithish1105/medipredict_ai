/**
 * MediPredict AI — Node.js Startup Wrapper
 * ========================================
 * Spawns the python backend process automatically. 
 * This enables deployment on standard PaaS environments (like Heroku, 
 * Render, or PM2) that expect a Node.js process to run the app.
 */

const { spawn } = require('child_process');
const path = require('path');
const os = require('os');

// Path to the Flask backend app.py
const backendAppPath = path.join(__dirname, 'project', 'backend', 'app.py');

// Use 'python3' on non-windows platforms, fallback to 'python' if needed
const isWindows = os.platform() === 'win32';
const pythonCmd = isWindows ? 'python' : 'python3';

console.log(`[MediPredict AI] Launching Flask server via Node.js wrapper...`);
console.log(`[MediPredict AI] Command: ${pythonCmd} ${backendAppPath}`);

// Spawn python process and forward stdout/stderr to Node.js logs
const flaskProcess = spawn(pythonCmd, [backendAppPath], {
    stdio: 'inherit',
    shell: true,
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
});

flaskProcess.on('error', (error) => {
    console.error(`[MediPredict AI] [Error] Failed to start Flask server:`, error);
});

flaskProcess.on('close', (code) => {
    console.log(`[MediPredict AI] Flask server process exited with code ${code}`);
    process.exit(code);
});

// Handle termination signals gracefully
const stopServer = () => {
    console.log(`[MediPredict AI] Stopping Flask server...`);
    flaskProcess.kill('SIGINT');
};

process.on('SIGTERM', stopServer);
process.on('SIGINT', stopServer);
