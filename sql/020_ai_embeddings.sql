CREATE TABLE IF NOT EXISTS ai.document_embedding (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  contents   text,
  metadata   jsonb,
  embedding  vector(1536),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS document_embedding_hnsw
  ON ai.document_embedding USING hnsw (embedding vector_cosine_ops);
