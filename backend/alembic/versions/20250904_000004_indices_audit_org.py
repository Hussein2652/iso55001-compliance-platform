"""add indices, audit fields, and org

Revision ID: 000004_indices_audit_org
Revises: 000003_management_reviews
Create Date: 2025-09-04 00:40:00

"""
from alembic import op
import sqlalchemy as sa


revision = '000004_indices_audit_org'
down_revision = '000003_management_reviews'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Organizations
    op.create_table(
        'organizations',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('created_at', sa.String(), nullable=False),
    )

    # Add fields to main tables
    for table in ['assessments', 'audits', 'nonconformities', 'management_reviews', 'attachments']:
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column('org_id', sa.String(), nullable=True))
            batch.add_column(sa.Column('created_by', sa.String(), nullable=True))
            batch.add_column(sa.Column('updated_by', sa.String(), nullable=True))
            batch.add_column(sa.Column('request_id', sa.String(), nullable=True))
            batch.create_index(f'ix_{table}_org_id', ['org_id'])
            batch.create_index(f'ix_{table}_created_at', ['created_at'])

    # Targeted indices
    op.create_index('ix_assessments_clause_id', 'assessments', ['clause_id'])
    op.create_index('ix_assessments_status_due', 'assessments', ['status', 'due_date'])
    op.create_index('ix_audits_status', 'audits', ['status'])
    op.create_index('ix_nonconformities_status_due', 'nonconformities', ['status', 'due_date'])
    op.create_index('ix_nonconformities_severity', 'nonconformities', ['severity'])


def downgrade() -> None:
    for table in ['assessments', 'audits', 'nonconformities', 'management_reviews', 'attachments']:
        with op.batch_alter_table(table) as batch:
            try:
                batch.drop_index(f'ix_{table}_org_id')
                batch.drop_index(f'ix_{table}_created_at')
            except Exception:
                pass
            for col in ['org_id', 'created_by', 'updated_by', 'request_id']:
                try:
                    batch.drop_column(col)
                except Exception:
                    pass
    for name in [
        'ix_assessments_clause_id', 'ix_assessments_status_due', 'ix_audits_status',
        'ix_nonconformities_status_due', 'ix_nonconformities_severity'
    ]:
        try:
            op.drop_index(name)
        except Exception:
            pass
    op.drop_table('organizations')

