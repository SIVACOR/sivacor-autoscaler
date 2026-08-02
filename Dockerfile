# The controller, packaged so `deploy-sivacor` can run it as a stack service
# instead of a checkout plus a terminal on the manager.
#
# Deliberately thin. Everything this needs at runtime arrives as environment or as
# a bind mount:
#
#   worker-cloud-init.sh   bind-mounted from deploy-sivacor. NOT baked in -- it is a
#                          deployment artifact, it changes far more often than this
#                          code, and baking it would create two artifacts that must
#                          agree. That is exactly the argument P2.1 used to reject a
#                          Packer image, and it applies here for the same reason.
#   clouds.yaml            bind-mounted read-only; the only way this process can
#                          create or delete instances.
#   secrets                environment, from the stack's .env (see ENVIRONMENT.md).
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/SIVACOR/sivacor-autoscaler"
LABEL org.opencontainers.image.description="On-demand OpenStack worker instances for SIVACOR"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY sivacor_autoscaler ./sivacor_autoscaler
RUN pip install --no-cache-dir .

# Runs as a non-root uid with no write access to anything that matters. This
# container creates and destroys VMs; it has no reason to be root, and nothing it
# touches is on a host filesystem it owns. Both bind mounts are read-only.
RUN useradd --create-home --uid 1000 autoscaler
USER autoscaler

# No CMD arguments: `-v` raises this package's own logger to DEBUG, and while the
# library loggers are pinned at INFO (so request bodies containing MASTER_KEY_HEX and
# REDIS_PASSWORD can no longer be logged -- see the plan's secret-exposure section),
# `docker service logs` is a shared sink and the default should stay quiet. Add `-v`
# in the stack's `command:` when debugging, and treat the output as secret-bearing
# until you have checked it.
ENTRYPOINT ["sivacor-autoscaler"]
