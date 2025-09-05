"""migrate string dates to proper types (postgres)

Revision ID: 000011_datetime_migrate
Revises: 000010_planning_context_ops
Create Date: 2025-09-05 00:50:00

"""
from alembic import op
import sqlalchemy as sa


revision = '000011_datetime_migrate'
down_revision = '000010_planning_context_ops'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        return
    # Helper to alter column type with USING clause
    def _alter(table, column, newtype):
        try:
            op.execute(f'ALTER TABLE {table} ALTER COLUMN {column} TYPE {newtype} USING NULLIF({column}, \"\")::{newtype}')
        except Exception:
            pass
    # created_at / updated_at -> timestamptz
    for table in ['assessments','attachments','audits','nonconformities','management_reviews','organizations','documents','ai_documents','ai_runs','kpi_definitions','kpi_results','samps','am_objectives','am_plans','risks','opportunities','context_issues','stakeholders','policies','role_map','leadership_attestations','assets','work_orders','providers','improvements']:
        _alter(table, 'created_at', 'timestamptz')
        _alter(table, 'updated_at', 'timestamptz')
    # date-like fields -> date
    for table, cols in [
        ('assessments',['due_date']),
        ('attachments',['retention_until']),
        ('audits',['scheduled_date','completed_date']),
        ('nonconformities',['due_date','closed_date','verified_on']),
        ('management_reviews',['period_start','period_end','meeting_date']),
        ('documents',['effective_date','next_review','retention_until']),
        ('kpi_results',['period']),
        ('samps',['period_start','period_end']),
        ('am_objectives',['due_date']),
        ('risks',['due_date']),
        ('work_orders',['sla_due']),
        ('policies',['effective_date']),
        ('leadership_attestations',['date']),
    ]:
        for col in cols:
            _alter(table, col, 'date')


def downgrade() -> None:
    # no-op: keeping improved types
    pass

