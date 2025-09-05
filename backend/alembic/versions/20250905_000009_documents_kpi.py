"""documents and kpis

Revision ID: 000009_documents_kpi
Revises: 000008_nc_capa_v2
Create Date: 2025-09-05 00:28:00

"""
from alembic import op
import sqlalchemy as sa


revision = '000009_documents_kpi'
down_revision = '000008_nc_capa_v2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'documents',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('version', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('approver_id', sa.String(), nullable=True),
        sa.Column('effective_date', sa.String(), nullable=True),
        sa.Column('next_review', sa.String(), nullable=True),
        sa.Column('s3_key', sa.String(), nullable=True),
        sa.Column('retention_until', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
        sa.Column('created_by', sa.String(), nullable=True),
        sa.Column('updated_by', sa.String(), nullable=True),
        sa.Column('request_id', sa.String(), nullable=True),
    )
    op.create_index('ix_documents_org', 'documents', ['org_id'])
    op.create_index('ix_documents_created', 'documents', ['created_at'])

    op.create_table(
        'kpi_definitions',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('org_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('method', sa.String(), nullable=True),
        sa.Column('frequency', sa.String(), nullable=True),
        sa.Column('target', sa.String(), nullable=True),
        sa.Column('owner_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )
    op.create_index('ix_kpi_def_org', 'kpi_definitions', ['org_id'])

    op.create_table(
        'kpi_results',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('kpi_id', sa.String(), nullable=False),
        sa.Column('period', sa.String(), nullable=False),
        sa.Column('value', sa.String(), nullable=True),
        sa.Column('evaluation', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=False),
    )
    op.create_index('ix_kpi_results_kpi_period', 'kpi_results', ['kpi_id', 'period'])


def downgrade() -> None:
    try:
        op.drop_index('ix_kpi_results_kpi_period')
    except Exception:
        pass
    op.drop_table('kpi_results')
    try:
        op.drop_index('ix_kpi_def_org')
    except Exception:
        pass
    op.drop_table('kpi_definitions')
    try:
        op.drop_index('ix_documents_created')
        op.drop_index('ix_documents_org')
    except Exception:
        pass
    op.drop_table('documents')

