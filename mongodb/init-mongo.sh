#!/bin/bash
set -e

echo "Starting MongoDB initialization..."

# This script runs in the Docker init phase: standalone mode, no auth enforcement.
# DO NOT call rs.initiate() here - replication is not enabled yet.
# DO NOT create the admin user here - use MONGO_INITDB_ROOT_USERNAME/PASSWORD in
# docker-compose so the image creates it and uses those credentials for rs.initiate()
# after the full restart.

MONGOT_PWD=$(cat /etc/mongodb-passwords/mongot_pwfile)
FIRMWARE_PWD=$(cat /etc/mongodb-passwords/firmware_pwfile)

echo "Creating application users..."
mongosh --eval "
  const adminDb = db.getSiblingDB('admin');

  function createUserSafe(db, spec) {
    try {
      db.createUser(spec);
      print('User ' + spec.user + ' created successfully');
    } catch (error) {
      if (error.code === 51003 || error.code === 11000) {
        print('User ' + spec.user + ' already exists, skipping');
      } else {
        throw error;
      }
    }
  }

  createUserSafe(adminDb, {
    user: 'firmware_user',
    pwd: '$FIRMWARE_PWD',
    roles: [
      { role: 'readWrite', db: 'firmware_analysis' }
    ]
  });

  createUserSafe(adminDb, {
    user: 'mongotUser',
    pwd: '$MONGOT_PWD',
    roles: [{ role: 'searchCoordinator', db: 'admin' }]
  });
"

echo "MongoDB initialization completed."
