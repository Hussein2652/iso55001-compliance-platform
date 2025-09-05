"""foreign keys and extra indexes (postgres)

Revision ID: 000012_fks_indexes
Revises: 000011_datetime_migrate
Create Date: 2025-09-05 00:56:00

"""
from alembic import op
import sqlalchemy as sa


revision = '000012_fks_indexes'
down_revision = '000011_datetime_migrate'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        return
    try:
        op.create_foreign_key('fk_attachments_assessment', 'attachments', 'assessments', ['assessment_id'], ['id'], ondelete='CASCADE')
    except Exception:
        pass
    try:
        op.create_foreign_key('fk_nc_audit', 'nonconformities', 'audits', ['audit_id'], ['id'], ondelete='SET NULL')
    except Exception:
        pass
    try:
        op.create_foreign_key('fk_assessment_clause', 'assessments', 'clauses', ['clause_id'], ['clause_id'])
    except Exception:
        pass
    # Composite indexes
    for tbl in ['assessments','audits','nonconformities','management_reviews','attachments','documents']:
        try:
            op.create_index(f'ix_{tbl}_org_created', tbl, ['org_id','created_at'])
        except Exception:
            pass


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        return
    for name in ['fk_attachments_assessment','fk_nc_audit','fk_assessment_clause']:
        try:
            op.drop_constraint(name, None, type_='foreignkey')
        except Exception:
            pass
    for tbl in ['assessments','audits','nonconformities','management_reviews','attachments','documents']:
        try:
            op.drop_index(f'ix_{tbl}_org_created')
        except Exception:
            pass

