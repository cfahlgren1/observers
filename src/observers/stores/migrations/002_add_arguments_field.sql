ALTER TABLE openai_records 
ADD COLUMN IF NOT EXISTS arguments JSON;
