#!/usr/bin/env bash
python manage.py makemigrations --noinput
python manage.py migrate --run-syncdb
# python manage.py collectstatic --noinput

# Create superuser if it doesn't exist
echo "Creating superuser..."
DJANGO_SUPERUSER_USERNAME=root DJANGO_SUPERUSER_EMAIL=root@example.com DJANGO_SUPERUSER_PASSWORD=root python manage.py createsuperuser --noinput || true

# Start gunicorn in the background
gunicorn autoweb.wsgi:application --bind 0.0.0.0:8085 --timeout 0
