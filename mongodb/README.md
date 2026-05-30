# MongoDB Setup

## Prerequisites

Run `generate-secrets.sh` to create a `pwfile` and `keyfile`. This also generates a password for `firmware_user` that you will need to add to your `.env`.

```bash
./generate-secrets.sh
```

## TLS Configuration

### With Let's Encrypt (recommended)

1. Obtain a certificate for your domain (e.g. via certbot).

2. Combine the private key and full chain into a single PEM file that MongoDB requires:

   ```bash
   cat /etc/letsencrypt/live/<your-domain>/privkey.pem \
       /etc/letsencrypt/live/<your-domain>/fullchain.pem \
       > ./tls/mongod.pem
   ```

3. Mount the entire letsencrypt directory in `docker-compose.yml` (already configured) so Docker correctly follows symlinks:

   ```yaml
   - /etc/letsencrypt:/etc/letsencrypt:ro
   ```

4. Update the cert paths in `mongod.conf` to match your domain.

**Cert renewal:** Let's Encrypt certs expire every 90 days. After each renewal, regenerate `mongod.pem` and restart the container. Add a certbot deploy hook to automate this:

```bash
# /etc/letsencrypt/renewal-hooks/deploy/restart-mongodb.sh
cat /etc/letsencrypt/live/<your-domain>/privkey.pem \
    /etc/letsencrypt/live/<your-domain>/fullchain.pem \
    > /path/to/mongodb/tls/mongod.pem
docker restart mongod-community
```

### With self-signed certificates

1. Generate a CA and server cert:

   ```bash
   mkdir -p tls
   openssl req -x509 -newkey rsa:4096 -keyout tls/ca.key -out tls/ca.pem -days 365 -nodes
   openssl req -newkey rsa:4096 -keyout tls/server.key -out tls/server.csr -nodes
   openssl x509 -req -in tls/server.csr -CA tls/ca.pem -CAkey tls/ca.key -CAcreateserial -out tls/server.crt -days 365
   cat tls/server.key tls/server.crt > tls/mongod.pem
   ```

2. The `./tls/` directory is already mounted in `docker-compose.yml`.

### Without TLS

Comment out the `tls:` block in `mongod.conf` and the TLS volume mounts in `docker-compose.yml`.

## Starting the stack

```bash
docker compose up -d
```
