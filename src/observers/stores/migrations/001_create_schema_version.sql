CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    migration_name VARCHAR,
    checksum VARCHAR
);

-- Initialize with version 0 if table is empty
INSERT INTO schema_version (version, migration_name) 
SELECT 0, 'initial' 
WHERE NOT EXISTS (SELECT 1 FROM schema_version);
