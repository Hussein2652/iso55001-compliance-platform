"""initial

Revision ID: 000001_initial
Revises: 
Create Date: 2025-09-04 00:00:01

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '000001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'clauses',
        sa.Column('clause_id', sa.String(), primary_key=True),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('summary', sa.String(), nullable=False),
    )
    op.create_table(
        'assessments',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('clause_id', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('evidence', sa.String()),
        sa.Column('owner', sa.String()),
        sa.Column('due_date', sa.String()),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )
    op.create_table(
        'attachments',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('assessment_id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('content_type', sa.String()),
        sa.Column('size', sa.Integer()),
        sa.Column('stored_path', sa.String(), nullable=False),
        sa.Column('created_at', sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('attachments')
    op.drop_table('assessments')
    op.drop_table('clauses')

