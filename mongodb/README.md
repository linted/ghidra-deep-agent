# MongoDB Setup

## Prerequisites

Run `generate-secrets.sh` to create a `pwfile` and `keyfile`. This also generates a password for `firmware_user` that you will need to add to your `.env`.

```bash
./generate-secrets.sh
```

## TLS Configuration

Both mongod and mongot need TLS certificates. mongod serves external client connections; mongot serves mongod's internal gRPC search traffic. They use separate certificates but must share the same CA so each can verify the other.

### With Let's Encrypt (recommended)

1. Obtain a certificate for your domain (e.g. via certbot).

2. Combine the private key and full chain into a single PEM file and write it directly into the letsencrypt live directory:

   ```bash
   cat /etc/letsencrypt/live/<your-domain>/privkey.pem \
       /etc/letsencrypt/live/<your-domain>/fullchain.pem \
       > /etc/letsencrypt/live/<your-domain>/mongod.pem
   ```

3. Generate a separate certificate for mongot signed by the same CA. Since `mongot.search-community` is an internal Docker alias (not publicly resolvable), it cannot get a Let's Encrypt cert directly — sign it with your own CA and bundle that CA into `ca_pem` alongside the Let's Encrypt chain:

   ```bash
   # Generate an internal CA (skip if you already have one)
   openssl req -x509 -newkey rsa:4096 -keyout tls/internal-ca.key -out tls/internal-ca.pem \
       -days 3650 -nodes -subj "/CN=Internal CA"

   # Generate and sign the mongot certificate
   openssl req -newkey rsa:4096 -keyout tls/mongot.key -out tls/mongot.csr -nodes \
       -subj "/CN=mongot.search-community"
   openssl x509 -req -in tls/mongot.csr -CA tls/internal-ca.pem -CAkey tls/internal-ca.key \
       -CAcreateserial -out tls/mongot.crt -days 365 \
       -extfile <(printf "subjectAltName=DNS:mongot.search-community")
   cat tls/mongot.key tls/mongot.crt > tls/mongot.pem

   # Bundle Let's Encrypt chain + internal CA into a single ca_pem so mongod trusts both
   cat /etc/letsencrypt/live/<your-domain>/fullchain.pem tls/internal-ca.pem > tls/ca.pem
   ```

4. Replace `mongod.search-community` with your domain in `docker-compose.yml` and `mongot.conf`. This must match the cert's CN/SAN so mongot can verify the TLS connection to mongod:

   ```bash
   sed -i 's/mongod.search-community/<your-domain>/g' docker-compose.yml mongot.conf
   ```

5. Update the `mongod_pem` and `ca_pem` file paths in the `secrets:` section of `docker-compose.yml`:

   ```yaml
   mongod_pem:
     file: /etc/letsencrypt/live/<your-domain>/mongod.pem
   mongot_pem:
     file: ./tls/mongot.pem
   ca_pem:
     file: ./tls/ca.pem   # bundled Let's Encrypt chain + internal CA
   ```

**Cert renewal:** Let's Encrypt certs expire every 90 days. Add a certbot deploy hook to regenerate `mongod.pem` and restart the containers:

```bash
# /etc/letsencrypt/renewal-hooks/deploy/restart-mongodb.sh
cat /etc/letsencrypt/live/<your-domain>/privkey.pem \
    /etc/letsencrypt/live/<your-domain>/fullchain.pem \
    > /etc/letsencrypt/live/<your-domain>/mongod.pem
# Re-bundle ca_pem (fullchain changes on renewal)
cat /etc/letsencrypt/live/<your-domain>/fullchain.pem tls/internal-ca.pem > tls/ca.pem
docker restart mongod-community mongot-community-pupr
```

### With self-signed certificates

1. Generate a CA and separate server certs for mongod and mongot:

   ```bash
   mkdir -p tls

   # CA
   openssl req -x509 -newkey rsa:4096 -keyout tls/ca.key -out tls/ca.pem \
       -days 3650 -nodes -subj "/CN=Local CA"

   # mongod certificate (CN must match your mongod hostname / domain)
   openssl req -newkey rsa:4096 -keyout tls/mongod.key -out tls/mongod.csr -nodes \
       -subj "/CN=mongod.search-community"
   openssl x509 -req -in tls/mongod.csr -CA tls/ca.pem -CAkey tls/ca.key \
       -CAcreateserial -out tls/mongod.crt -days 365 \
       -extfile <(printf "subjectAltName=DNS:mongod.search-community")
   cat tls/mongod.key tls/mongod.crt > tls/mongod.pem

   # mongot certificate
   openssl req -newkey rsa:4096 -keyout tls/mongot.key -out tls/mongot.csr -nodes \
       -subj "/CN=mongot.search-community"
   openssl x509 -req -in tls/mongot.csr -CA tls/ca.pem -CAkey tls/ca.key \
       -CAcreateserial -out tls/mongot.crt -days 365 \
       -extfile <(printf "subjectAltName=DNS:mongot.search-community")
   cat tls/mongot.key tls/mongot.crt > tls/mongot.pem
   ```

2. The `./tls/` directory is already mounted in `docker-compose.yml`.

### Without TLS

Comment out the `tls:` block in `mongod.conf`, set `server.grpc.tls.mode: "disabled"` in `mongot.conf`, and remove the TLS secrets and volume mounts from `docker-compose.yml`.

## Starting the stack

```bash
docker compose up -d
```
