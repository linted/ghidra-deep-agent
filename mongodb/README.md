# MongoDB Setup

## Prerequisites

Run `generate-secrets.sh` to create a `pwfile` and `keyfile`. This also generates a password for `firmware_user` that you will need to add to your `.env`.

```bash
./generate-secrets.sh
```

## TLS Configuration

### With Let's Encrypt (recommended)

1. Obtain a certificate for your domain (e.g. via certbot).

2. Combine the private key and full chain into a single PEM file and write it directly into the letsencrypt live directory:

   ```bash
   cat /etc/letsencrypt/live/<your-domain>/privkey.pem \
       /etc/letsencrypt/live/<your-domain>/fullchain.pem \
       > /etc/letsencrypt/live/<your-domain>/mongod.pem
   ```

3. Replace `mongod.search-community` with your domain in `docker-compose.yml` and `mongot.conf`. This must match the cert's CN/SAN so mongot can verify the TLS connection to mongod:

   ```bash
   sed -i 's/mongod.search-community/<your-domain>/g' docker-compose.yml mongot.conf
   ```

4. Update the `mongod_pem` and `ca_pem` file paths in the `secrets:` section of `docker-compose.yml`:

   ```yaml
   mongod_pem:
     file: /etc/letsencrypt/live/<your-domain>/mongod.pem
   ca_pem:
     file: /etc/letsencrypt/live/<your-domain>/fullchain.pem
   ```

**Cert renewal:** Let's Encrypt certs expire every 90 days. Add a certbot deploy hook to regenerate `mongod.pem` and restart the containers:

```bash
# /etc/letsencrypt/renewal-hooks/deploy/restart-mongodb.sh
cat /etc/letsencrypt/live/<your-domain>/privkey.pem \
    /etc/letsencrypt/live/<your-domain>/fullchain.pem \
    > /etc/letsencrypt/live/<your-domain>/mongod.pem
docker restart mongod-community mongot-community-pupr
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

Comment out the `tls:` block in `mongod.conf`, set `tls: false` in `mongot.conf`, and remove the TLS volume mounts from `docker-compose.yml`.

## Starting the stack

```bash
docker compose up -d
```
