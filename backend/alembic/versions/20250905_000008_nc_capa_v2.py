"""nc/capa v2 columns

Revision ID: 000008_nc_capa_v2
Revises: 000007_ai_rag
Create Date: 2025-09-05 00:20:00

"""
from alembic import op
import sqlalchemy as sa


revision = '000008_nc_capa_v2'
down_revision = '000007_ai_rag'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('nonconformities') as batch:
        batch.add_column(sa.Column('containment', sa.String(), nullable=True))
        batch.add_column(sa.Column('root_cause', sa.String(), nullable=True))
        batch.add_column(sa.Column('preventive_action', sa.String(), nullable=True))
        batch.add_column(sa.Column('verification_method', sa.String(), nullable=True))
        batch.add_column(sa.Column('verified_by', sa.String(), nullable=True))
        batch.add_column(sa.Column('verified_on', sa.String(), nullable=True))
        batch.add_column(sa.Column('state', sa.String(), nullable=True))
    # Helpful composite index
    try:
        op.create_index('ix_nonconformities_org_created', 'nonconformities', ['org_id', 'created_at'])
    except Exception:
        pass


def downgrade() -> None:
    try:
        op.drop_index('ix_nonconformities_org_created')
    except Exception:
        pass
    with op.batch_alter_table('nonconformities') as batch:
        for col in ['state','verified_on','verified_by','verification_method','preventive_action','root_cause','containment']:
            try:
                batch.drop_column(col)
            except Exception:
                pass

