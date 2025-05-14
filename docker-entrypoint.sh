set -e

redis-server --daemonize yes

python manage.py migrate

python manage.py collectstatic --noinput

exec "$@"
