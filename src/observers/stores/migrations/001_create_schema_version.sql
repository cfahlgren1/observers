CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    migration_name VARCHAR,
    checksum VARCHAR
);

CREATE TABLE IF NOT EXISTS openai_records (
    id VARCHAR PRIMARY KEY,
    model VARCHAR,
    timestamp TIMESTAMP,
    messages JSON,
    assistant_message TEXT,
    completion_tokens INTEGER,
    prompt_tokens INTEGER,
    total_tokens INTEGER,
    finish_reason VARCHAR,
    tool_calls JSON,
    function_call JSON,
    tags VARCHAR[],
    properties JSON,
    error VARCHAR,
    raw_response JSON,
    arguments JSON
);

-- Initialize with version 0 if table is empty
INSERT INTO schema_version (version, migration_name) 
SELECT 0, 'initial' 
WHERE NOT EXISTS (SELECT 1 FROM schema_version);
