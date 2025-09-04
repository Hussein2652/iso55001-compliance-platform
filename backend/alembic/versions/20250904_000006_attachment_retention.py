"""attachment retention and hash fields

Revision ID: 000006_attachment_retention
Revises: 000004_indices_audit_org
Create Date: 2025-09-04 01:10:00

"""
from alembic import op
import sqlalchemy as sa


revision = '000006_attachment_retention'
down_revision = '000004_indices_audit_org'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('attachments') as batch:
        batch.add_column(sa.Column('sha256', sa.String(), nullable=True))
        batch.add_column(sa.Column('retention_hold', sa.String(), nullable=True))
        batch.add_column(sa.Column('retention_until', sa.String(), nullable=True))
        batch.add_column(sa.Column('disposition', sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('attachments') as batch:
        try:
            batch.drop_column('disposition')
            batch.drop_column('retention_until')
            batch.drop_column('retention_hold')
            batch.drop_column('sha256')
        except Exception:
            pass

