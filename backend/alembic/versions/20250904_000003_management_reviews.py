"""management reviews

Revision ID: 000003_management_reviews
Revises: 000002_audits_nonconformities
Create Date: 2025-09-04 00:20:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '000003_management_reviews'
down_revision = '000002_audits_nonconformities'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'management_reviews',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('period_start', sa.String()),
        sa.Column('period_end', sa.String()),
        sa.Column('meeting_date', sa.String()),
        sa.Column('participants', sa.String()),
        sa.Column('summary', sa.String()),
        sa.Column('decisions', sa.String()),
        sa.Column('actions', sa.String()),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('management_reviews')

