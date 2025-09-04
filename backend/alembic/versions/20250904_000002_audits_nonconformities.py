"""audits and nonconformities

Revision ID: 000002_audits_nonconformities
Revises: 000001_initial
Create Date: 2025-09-04 00:10:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '000002_audits_nonconformities'
down_revision = '000001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'audits',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('description', sa.String()),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('scheduled_date', sa.String()),
        sa.Column('completed_date', sa.String()),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )
    op.create_table(
        'nonconformities',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('audit_id', sa.Integer()),
        sa.Column('clause_id', sa.String()),
        sa.Column('severity', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('corrective_action', sa.String()),
        sa.Column('owner', sa.String()),
        sa.Column('due_date', sa.String()),
        sa.Column('closed_date', sa.String()),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('nonconformities')
    op.drop_table('audits')

