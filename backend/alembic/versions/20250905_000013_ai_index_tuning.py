"""ai embeddings index tuning

Revision ID: 000013_ai_index_tuning
Revises: 000012_fks_indexes
Create Date: 2025-09-05 01:02:00

"""
from alembic import op


revision = '000013_ai_index_tuning'
down_revision = '000012_fks_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        return
    try:
        op.execute('DROP INDEX IF EXISTS ai_embeddings_vec_idx')
    except Exception:
        pass
    op.execute('CREATE INDEX ai_embeddings_vec_idx ON ai_embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists=100)')


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        return
    try:
        op.execute('DROP INDEX IF EXISTS ai_embeddings_vec_idx')
    except Exception:
        pass
    op.execute('CREATE INDEX ai_embeddings_vec_idx ON ai_embeddings USING ivfflat (vector vector_cosine_ops)')

