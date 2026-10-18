"""Windows-only launcher; starts both APIs and three Streamlit browser windows.
Run only AFTER training models/phone_original.pt and models/phone_copied.pt.
"""
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [ROOT/'models'/'phone_original.pt', ROOT/'models'/'phone_copied.pt',
            ROOT/'demo_samples'/'sandal_watermarked.png']


def port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) != 0


def main():
    missing = [str(p) for p in REQUIRED if not p.is_file()]
    if missing:
        raise SystemExit('Train both Phone models first. Missing: ' + ', '.join(missing))
    if os.name != 'nt':
        raise SystemExit('This one-click launcher is for Windows. Use commands in README.md on other systems.')
    busy = [p for p in (8765, 8766, 8501, 8502, 8503) if not port_available(p)]
    if busy:
        raise SystemExit(f'Ports {busy} already in use. Stop earlier API/Streamlit runs first (Ctrl+C).')

    jobs = [
        ('ORIGINAL AI API', ['-m', 'uvicorn', 'phone_api:app', '--host', '127.0.0.1', '--port', '8765'], 'phone_original.pt'),
        ('COPIED AI API', ['-m', 'uvicorn', 'phone_api:app', '--host', '127.0.0.1', '--port', '8766'], 'phone_copied.pt'),
        ('ORIGINAL WINDOW', ['-m', 'streamlit', 'run', 'original_window.py', '--server.address', '127.0.0.1', '--server.port', '8501', '--server.headless', 'true'], None),
        ('COPIED WINDOW', ['-m', 'streamlit', 'run', 'copied_window.py', '--server.address', '127.0.0.1', '--server.port', '8502', '--server.headless', 'true'], None),
        ('DETECTION WINDOW', ['-m', 'streamlit', 'run', 'detection_window.py', '--server.address', '127.0.0.1', '--server.port', '8503', '--server.headless', 'true'], None),
    ]
    print('Starting separate console windows...')
    for title, args, filename in jobs:
        env = os.environ.copy()
        if filename:
            env['MODEL_FILE'] = filename
        command = [sys.executable, *args]
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
        print(f'{title}: PID {process.pid}')
    print('\nOpen these three URLs in three browser tabs/windows:')
    print('ORIGINAL AI: http://127.0.0.1:8501')
    print('COPIED AI:   http://127.0.0.1:8502')
    print('DETECTION:   http://127.0.0.1:8503')
    print('Close the launched console windows to stop the demo.')


if __name__ == '__main__':
    main()
