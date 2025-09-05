"""ai rag tables and pgvector

Revision ID: 000007_ai_rag
Revises: 000006_attachment_retention
Create Date: 2025-09-05 00:00:07

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '000007_ai_rag'
down_revision = '000006_attachment_retention'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name if bind else None

    # ai_documents
    op.create_table(
        'ai_documents',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('source_type', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('path', sa.String(), nullable=True),
        sa.Column('mime', sa.String(), nullable=True),
        sa.Column('hash', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
    )
    op.create_index('ix_ai_documents_org', 'ai_documents', ['org_id'])
    op.create_index('ix_ai_documents_created', 'ai_documents', ['created_at'])

    # ai_chunks
    op.create_table(
        'ai_chunks',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('document_id', sa.String(), sa.ForeignKey('ai_documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('ord', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=True),
    )
    op.create_index('ix_ai_chunks_doc_ord', 'ai_chunks', ['document_id', 'ord'])

    # ai_embeddings: vector type when Postgres, TEXT (JSON) fallback otherwise
    if dialect == 'postgresql':
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        op.execute(
            """
            CREATE TABLE ai_embeddings (
              id TEXT PRIMARY KEY,
              chunk_id TEXT REFERENCES ai_chunks(id) ON DELETE CASCADE,
              vector VECTOR(1024)
            )
            """
        )
        op.execute(
            "CREATE INDEX ai_embeddings_vec_idx ON ai_embeddings USING ivfflat (vector vector_cosine_ops)"
        )
    else:
        op.create_table(
            'ai_embeddings',
            sa.Column('id', sa.String(), primary_key=True),
            sa.Column('chunk_id', sa.String(), sa.ForeignKey('ai_chunks.id', ondelete='CASCADE'), nullable=False),
            sa.Column('vector', sa.Text(), nullable=False),  # JSON-encoded vector fallback
        )
        op.create_index('ix_ai_embeddings_chunk', 'ai_embeddings', ['chunk_id'])

    # ai_runs
    op.create_table(
        'ai_runs',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=True),
        sa.Column('task', sa.String(), nullable=False),
        sa.Column('model', sa.String(), nullable=True),
        sa.Column('temperature', sa.String(), nullable=True),
        sa.Column('top_p', sa.String(), nullable=True),
        sa.Column('seed', sa.String(), nullable=True),
        sa.Column('inputs_json', sa.Text(), nullable=True),
        sa.Column('retrieved_refs_json', sa.Text(), nullable=True),
        sa.Column('output_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
    )
    op.create_index('ix_ai_runs_org', 'ai_runs', ['org_id'])
    op.create_index('ix_ai_runs_created', 'ai_runs', ['created_at'])


def downgrade() -> None:
    # Drop in reverse dependency order
    try:
        op.drop_index('ix_ai_runs_created')
        op.drop_index('ix_ai_runs_org')
    except Exception:
        pass
    op.drop_table('ai_runs')

    bind = op.get_bind()
    dialect = bind.dialect.name if bind else None
    if dialect == 'postgresql':
        try:
            op.execute('DROP INDEX IF EXISTS ai_embeddings_vec_idx')
        except Exception:
            pass
        op.execute('DROP TABLE IF EXISTS ai_embeddings')
    else:
        try:
            op.drop_index('ix_ai_embeddings_chunk')
        except Exception:
            pass
        op.drop_table('ai_embeddings')

    try:
        op.drop_index('ix_ai_chunks_doc_ord')
    except Exception:
        pass
    op.drop_table('ai_chunks')

    try:
        op.drop_index('ix_ai_documents_created')
        op.drop_index('ix_ai_documents_org')
    except Exception:
        pass
    op.drop_table('ai_documents')

