#!/bin/bash
set -e

echo "Starting MongoDB initialization..."

# This script runs in the Docker init phase: standalone mode, no auth enforcement.
# DO NOT call rs.initiate() here - replication is not enabled yet.
# DO NOT create the admin user here - use MONGO_INITDB_ROOT_USERNAME/PASSWORD in
# docker-compose so the image creates it and uses those credentials for rs.initiate()
# after the full restart.

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
    pwd: 'f2970a49-95bc-412c-8983-8c69d0a01e01',
    roles: [
      { role: 'readWrite', db: 'firmware_analysis' }
    ]
  });

  createUserSafe(adminDb, {
    user: 'mongotUser',
    pwd: 'b0533610-6d55-4a78-bbbe-afaace6c1fb3',
    roles: [{ role: 'searchCoordinator', db: 'admin' }]
  });
"

echo "MongoDB initialization completed."
