# Use Ubuntu as the base image
FROM ubuntu:latest

# Set environment variables for PostgreSQL
ENV POSTGRES_USER=admin
ENV POSTGRES_PASSWORD=admin123
ENV POSTGRES_DB=mydatabase
# Set environment variables for pgAdmin
ENV PGADMIN_DEFAULT_EMAIL=admin@example.com
ENV PGADMIN_DEFAULT_PASSWORD=admin123
ENV MASTER_PASSWORD_REQUIRED=False
ENV PGADMIN_LISTEN_PORT=5050
ENV ALLOW_SAVE_PASSWORD=False

# Update and install required packages
RUN apt-get update && apt-get install -y \
    postgresql postgresql-contrib \
    wget curl sudo python3 python3-pip \
    python3-venv && apt-get clean

# Create a virtual environment for pgAdmin
RUN python3 -m venv /pgadmin4_venv && \
   /pgadmin4_venv/bin/pip install pgadmin4

# Create necessary directories
RUN mkdir -p /var/lib/postgresql/data
VOLUME /var/lib/postgresql/data
RUN mkdir -p /tmp/

#Locate default python version installed in venv and its directory \
RUN echo "#!/bin/bash" >> /tmp/find_first_dir.sh \
    && echo "VENV_PYTHON_DIR=\$(ls -d ./lib/*/ | head -n 1 | sed 's|./lib/||' | sed 's|/||')" >> /tmp/find_first_dir.sh \
    && echo "echo \"VENV_PYTHON_DIR=\$VENV_PYTHON_DIR\"" >> /tmp/find_first_dir.sh

RUN chmod +x /tmp/find_first_dir.sh

# Run the script to set the FIRST_DIR environment variable
RUN . /tmp/find_first_dir.sh && \
    echo "Python dir: $VENV_PYTHON_DIR"

# Now you can use the $FIRST_DIR variable in subsequent commands
RUN ls ./lib/$VENV_PYTHON_DIR/site-packages/pgadmin4/setup.py || echo "setup.py not found in the expected location"

# Clean up the temporary script (optional)
RUN rm /tmp/find_first_dir.sh
# Set up pgAdmin configuration directory and inject environment variables

RUN mkdir -p /root/.pgadmin && \
#   echo "SERVER_MODE = False" > /root/.pgadmin/config_local.py && \
#   echo "DEFAULT_SERVER = '0.0.0.0'" >> /root/.pgadmin/config_local.py && \
   echo "DEFAULT_USER = '${PGADMIN_DEFAULT_EMAIL}'" >> /pgadmin4_venv/lib/python3.12/site-packages/pgadmin4/config.py && \
   echo "DEFAULT_PASSWORD = '${PGADMIN_DEFAULT_PASSWORD}'" >> /pgadmin4_venv/lib/python3.12/site-packages/pgadmin4/config.py && \
    echo "MASTER_PASSWORD_REQUIRED = '${MASTER_PASSWORD_REQUIRED}'" >> /pgadmin4_venv/lib/python3.12/site-packages/pgadmin4/config.py && \
    echo "PGADMIN_LISTEN_PORT = '${PGADMIN_LISTEN_PORT}'" >> /pgadmin4_venv/lib/python3.12/site-packages/pgadmin4/config.py && \
    echo "ALLOW_SAVE_PASSWORD = '${ALLOW_SAVE_PASSWORD}'" >> /pgadmin4_venv/lib/python3.12/site-packages/pgadmin4/config.py


# Expose PostgreSQL and pgAdmin ports
EXPOSE 5432 ${PGADMIN_LISTEN_PORT}

# Start PostgreSQL and pgAdmin
#CMD service postgresql start && /pgadmin4_venv/bin/pgadmin4
CMD ["sh", "-c", "service postgresql start && /pgadmin4_venv/bin/pgadmin4"]

##Build the Docker Image
#docker build -t ubuntu_pgadmin_postgres .
#
##Run the Container
#docker run -p 5432:5432 -p 5050:5050 -d ubuntu_pgadmin_postgres
