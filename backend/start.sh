#!/usr/bin/env bash

# Start Anvil blockchain server as background process
if [ -f "/usr/src/app/anvil_state.json" ]; then
    echo "Starting Anvil with saved state..."
    nohup anvil --load-state /usr/src/app/anvil_state.json --host 0.0.0.0 > /usr/src/app/anvil.log 2>&1 &
else
    echo "Starting Anvil with default state..."
    nohup anvil > /usr/src/app/anvil.log 2>&1 &
fi

# Wait for Anvil to start
sleep 5

# Check if Anvil is running
if pgrep -f "anvil" > /dev/null; then
    echo "Anvil blockchain server started successfully"
else
    echo "Warning: Anvil blockchain server failed to start"
fi

python manage.py makemigrations --noinput
python manage.py migrate --run-syncdb
# gunicorn autoweb.wsgi:application --bind 0.0.0.0:8000 --timeout 0
daphne  -b 0.0.0.0 -p 8000 -t 0 autoweb.asgi:application